"""Convert the manually annotated workbook locally; never print annotation text."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def workbook_rows(path):
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(t.text or '' for t in item.findall('.//m:t', ns))
                      for item in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', ns)]
        tree = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        rows = []
        for row in tree.findall('.//m:sheetData/m:row', ns):
            values = {}
            for cell in row.findall('m:c', ns):
                v = cell.find('m:v', ns)
                value = v.text if v is not None else ''
                if cell.get('t') == 's':
                    value = shared[int(value)]
                elif cell.get('t') == 'inlineStr':
                    value = ''.join(t.text or '' for t in cell.findall('.//m:t', ns))
                values[re.sub(r'\d', '', cell.get('r'))] = value or ''
            rows.append(values)
    return rows


def prepare(folder, old_manifest, freeze, output):
    folder, output = Path(folder), Path(output)
    old = json.loads(Path(old_manifest).read_text(encoding='utf-8'))['videos']
    split = json.loads(Path(freeze).read_text(encoding='utf-8'))
    if len(old) != len(split['splits']):
        raise ValueError('Previous split and manifest differ')
    by_hash = {}
    for row, label in zip(old, split['splits']):
        record = by_hash.setdefault(row['sha256'], {'split': label, 'binary_label': row['source']['binary_label'], 'anchors': []})
        if record['split'] != label or record['binary_label'] != row['source']['binary_label']:
            raise ValueError('Conflicting duplicate split/label')
        t = row['source']['timestamp_sec']
        if t is not None:
            record['anchors'].append(float(t))
    files = {p.stem.casefold(): p for p in (folder/'videos').rglob('*') if p.suffix.lower() in {'.mov', '.mp4'}}
    prepared = {}
    rows = workbook_rows(folder/'open_labels.xlsx')
    if not rows or not {'A', 'B', 'C', 'D'} <= set(rows[0]):
        raise ValueError('Expected filename, timestamp, binary label, annotation columns')
    for row_number,row in enumerate(rows[1:],start=2):
        if not row.get('A'):
            continue
        name = Path(row['A'].strip().replace('\\', '/')).stem.casefold()
        path = files.get(name)
        if path is None:
            raise ValueError('An annotation row has no matching video')
        sha = digest(path)
        if sha not in by_hash:
            raise ValueError('Video is absent from the frozen split')
        label = {'нарушение': 1, 'без нарушения': 0}.get(row.get('C','').strip().lower())
        if label is None or label != by_hash[sha]['binary_label']:
            raise ValueError('Manual and organizer labels disagree; review locally')
        annotation = row.get('D', '').strip()
        if not annotation:
            raise ValueError('Missing manual annotation')
        record = prepared.setdefault(sha, {'path': str(path.resolve()), 'sha256': sha,
            **by_hash[sha], 'annotations': [], 'source_rows': []})
        if annotation not in record['annotations']:
            record['annotations'].append(annotation)
        record['source_rows'].append(row_number)
    if len(prepared) != len(by_hash):
        raise ValueError('Annotations do not cover every unique video')
    result = {'schema_version': 1, 'workbook_sha256': digest(folder/'open_labels.xlsx'),
        'split_freeze_sha256': digest(freeze), 'privacy': 'local only',
        'split_note': 'Previous split preserved. Test already reported for earlier CPU baseline; not a pristine new benchmark.',
        'grouping': 'exact duplicates grouped; session independence unverified',
        'videos': list(prepared.values())}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    output.chmod(0o600)
    counts = Counter(v['split'] for v in result['videos'])
    report = {'unique_videos': len(prepared), 'annotation_rows': len(rows)-1,
        'splits': dict(counts), 'workbook_sha256': result['workbook_sha256'],
        'annotation_lengths': {'min': min(len(a) for v in prepared.values() for a in v['annotations']),
                               'max': max(len(a) for v in prepared.values() for a in v['annotations'])}}
    output.with_suffix('.summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--old-manifest',required=True)
    p.add_argument('--freeze',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    prepare(a.folder,a.old_manifest,a.freeze,a.output)
