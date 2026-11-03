"""Bind Material8 to its render target and flag clear qualitative VLM conflicts."""

import copy
import math
from pathlib import Path

from PIL import Image, ImageOps

from experiments.sdxl_mailbox_v1.data import NATURAL, read_json, sha256


POLICY = {'version': 'material_target_consistency/v1',
    'metallic_low_max': 0.2, 'metallic_high_min': 0.8,
    'roughness_low_max_for_high_label_conflict': 0.25,
    'roughness_high_min_for_low_label_conflict': 0.65,
    'scope': 'Conservative diagnostic thresholds, not calibrated physical class boundaries.'}


def _scalar(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Material target must contain finite metallic/roughness scalars in [0,1]')
    return value


def compare_material_target(semantic, roughness, metallic):
    """Only high-confidence, widely separated claims can block preparation."""
    roughness, metallic = _scalar(roughness), _scalar(metallic)
    findings = []
    for axis, scalar in (('metallicity', metallic), ('roughness', roughness)):
        value = semantic[axis]
        label, confidence = value['label'], value['confidence']
        labels = ('metallic', 'dielectric', 'mixed', 'uncertain') if axis == 'metallicity' else ('low', 'medium', 'high', 'uncertain')
        if label not in labels or confidence not in ('high', 'medium', 'low'):
            raise ValueError('invalid controlled Material class or confidence')
        if label in ('mixed', 'uncertain') or confidence != 'high' or (axis == 'roughness' and label == 'medium'):
            reason = ('mixed_surface_not_represented_by_one_scalar' if label == 'mixed' else
                      'intermediate_class_has_no_hard_boundary' if label == 'medium' else
                      'uncertain_or_nonhigh_confidence')
            findings.append({'axis': axis, 'status': 'inconclusive', 'reason': reason})
            continue
        conflict = ((axis == 'metallicity' and label == 'metallic' and scalar <= POLICY['metallic_low_max'])
            or (axis == 'metallicity' and label == 'dielectric' and scalar >= POLICY['metallic_high_min'])
            or (axis == 'roughness' and label == 'low' and scalar >= POLICY['roughness_high_min_for_low_label_conflict'])
            or (axis == 'roughness' and label == 'high' and scalar <= POLICY['roughness_low_max_for_high_label_conflict']))
        findings.append({'axis': axis, 'status': 'conflict' if conflict else 'no_clear_conflict',
                         'label': label, 'target': scalar, 'confidence': confidence})
    status = ('conflict' if any(row['status'] == 'conflict' for row in findings) else
              'inconclusive' if any(row['status'] == 'inconclusive' for row in findings) else 'no_clear_conflict')
    return {'status': status, 'policy': copy.deepcopy(POLICY), 'findings': findings,
            'physical_truth_verified': False, 'target_values_changed': False,
            'initializer_reselected': False, 'training_blocked': status == 'conflict'}


def _locked_json(path, locks):
    path = Path(path).resolve()
    if locks.get(str(path)) != sha256(path):
        raise ValueError(f'Material target evidence is not hash-locked: {path}')
    return read_json(path)


def _rgb(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        return image.size, image.tobytes()


def audit_material_target(manifest, rows, locks, natural_root=NATURAL):
    """Use validated source rows; append checked metadata locks for future rechecks."""
    root = Path(natural_root).resolve()
    protocol_path, render_path = root / 'plan/protocol.json', root / 'renders/manifest.json'
    protocol, rendered = _locked_json(protocol_path, locks), _locked_json(render_path, locks)
    source = protocol['sources']['mailbox']
    target = protocol['materials']['mailbox_material']
    material_path = Path(source['material']).resolve()
    extracted = _locked_json(material_path, locks)
    values = {key: _scalar(target[key]) for key in ('roughness', 'metallic')}
    if any(source[key] != value or extracted[key] != value for key, value in values.items()):
        raise ValueError('Material source, extracted scalar and rendering protocol targets differ')
    image = Path(manifest['source']['image']).resolve()
    source_image = Path(source['input']).resolve()
    if (sha256(image) != manifest['source']['image_sha256']
            or locks.get(str(source_image)) != sha256(source_image)
            or _rgb(image) != _rgb(source_image)):
        raise ValueError('VLM original photo and SuperMat source RGB differ')
    rendered_rows = [row for row in rendered['records'] if row['source_id'] == 'mailbox' and row['branch'] == 'material']
    ids = {row['id'] for row in rows}
    if (len(rows) != 8 or len(ids) != 8 or len(rendered_rows) != 8
            or {row['id'] for row in rendered_rows} != ids):
        raise ValueError('require the same eight Mailbox Material render and training rows')
    by_id = {row['id']: row for row in rendered_rows}
    for row in rows:
        rendered_row = by_id[row['id']]
        metadata_path = Path(rendered_row['metadata'])
        if not metadata_path.is_absolute():
            metadata_path = root / 'renders' / metadata_path
        metadata_path = metadata_path.resolve()
        digest = rendered_row['metadata_sha256']
        if (str(metadata_path) in locks and locks[str(metadata_path)] != digest) or sha256(metadata_path) != digest:
            raise ValueError('Material render metadata hash differs')
        locks[str(metadata_path)] = digest
        metadata = _locked_json(metadata_path, locks)
        if metadata != {key: value for key, value in rendered_row.items() if key not in ('metadata', 'metadata_sha256')}:
            raise ValueError('Material render metadata disagrees with render manifest')
        for record in (row, rendered_row, metadata):
            if (record['source_id'] != 'mailbox' or record['branch'] != 'material'
                    or record['material_id'] != 'mailbox_material'
                    or any(_scalar(record[key]) != value for key, value in values.items())):
                raise ValueError('Material8 staged, rendered or metadata M/R target differs')
        rendered_image = Path(rendered_row['image'])
        if not rendered_image.is_absolute():
            rendered_image = root / 'renders' / rendered_image
        if (row['image_sha256'] != rendered_row['image_sha256']
                or row['source_image'] != str(rendered_image.resolve())):
            raise ValueError('Material target row is attached to different training RGB')
        expected_shader = {'texture_image_nodes': 0, 'normal_bump_displacement': False,
                           'roughness_input_linked': False, 'metallic_input_linked': False,
                           'albedo_mode': 'uniform'}
        if any(rendered_row.get(key) != value for key, value in expected_shader.items()):
            raise ValueError('Material target must be uniform scalar M/R without texture or normal changes')
    report = compare_material_target(manifest['response']['material'], **values)
    report.update(schema='colorpeel_material_target_consistency/v1', numeric_target=values,
        vlm_material=copy.deepcopy(manifest['response']['material']), target_provenance_verified=True,
        original_image=str(image), original_image_sha256=manifest['source']['image_sha256'],
        supermat_input=str(source_image), supermat_input_sha256=locks[str(source_image)],
        source_rgb_equal=True, source_rgb_comparison='EXIF transpose and RGB decoding; PNG alpha ignored',
        material_file=str(material_path), material_file_sha256=locks[str(material_path)],
        selected_region=extracted.get('selected_region'), protocol=str(protocol_path),
        protocol_sha256=locks[str(protocol_path)], render_manifest=str(render_path),
        render_manifest_sha256=locks[str(render_path)], training_row_ids=sorted(ids))
    return report
