"""Create a PRIVATE review page next to local predictions. Do not upload this file."""
import argparse
import html
import json
import os
from pathlib import Path


def main(args):
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    rows={r['sha256']:r for r in manifest['videos']}
    folder=Path(args.evaluation);report=json.loads((folder/'report.json').read_text())
    parts=['<!doctype html><meta charset="utf-8"><title>Private local Qwen review</title>',
        '<style>body{max-width:1100px;margin:40px auto;font:17px system-ui;background:#101016;color:#eee}section{border:1px solid #555;padding:24px;margin:20px 0}pre{white-space:pre-wrap}a{color:#aaf}h2{font-size:20px}</style>',
        '<h1>Private local validation review</h1><p>Contains event annotations and model descriptions. Keep this file on this laptop. Model descriptions can be wrong. Check visible actors/actions, timing, rule prerequisites and exceptions against the video.</p>']
    parts.append('<p>Experiment: <strong>'+html.escape(folder.name)+'</strong>. Example numbers follow the frozen validation order.</p>')
    for i,record in enumerate(report['records']):
        row=rows[record['sha256']];private=folder/f'private-{i+1:02d}.json'
        prediction=json.loads(private.read_text()) if private.exists() else {'status':'No valid structured result'}
        parts += [f'<section><h2>Example {i+1}: expected {row["binary_label"]}, predicted {record.get("predicted","failed")}</h2>',
            '<a href="'+html.escape(Path(row['path']).as_uri(),quote=True)+'">Open original video locally</a>',
            '<h3>Human annotation</h3><pre>'+html.escape('\n'.join(row['annotations']))+'</pre>',
            '<h3>Qwen prediction — verify, do not treat as ground truth</h3><pre>'+html.escape(json.dumps(prediction,ensure_ascii=False,indent=2))+'</pre></section>']
    target=folder/'PRIVATE_REVIEW.html';target.write_text('\n'.join(parts),encoding='utf-8');os.chmod(target,0o600)
    print(json.dumps({'local_review_file':str(target),'examples':len(report['records'])}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--evaluation',required=True)
    main(p.parse_args())
