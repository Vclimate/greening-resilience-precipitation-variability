"""Remove production-note text from vector figure PDFs, preserving plotted data."""
import hashlib
import json
from pathlib import Path

import pymupdf

ROOT = Path(r'D:\papaer4\20260604返修补实验')
SOURCE = ROOT / 'output/QA01_unified22_20261005/figures_matlab_style'
OUT = ROOT / 'v15 - JRS/QA01_integrated_results_20261006/Figures_fullframe_white_v4/Comments_v5_20261008/figures'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prefixes = ['Panel b: original', 'TAC year =', 'Shading: calculated',
                'Significance: two-sided', 'Shading and brackets:']
    specs = [(1, 'Fig1_MATLAB_style_polished_v14_original_palette.pdf', 1),
             (2, 'Fig2_MATLAB_style_v5_HAC4.pdf', 3),
             (4, 'Fig4_MATLAB_style.pdf', 1)]
    records = []
    for number, name, expected in specs:
        source = SOURCE / name
        before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        doc = pymupdf.open(source)
        assert len(doc) == 1
        page = doc[0]
        blocks = page.get_text('blocks')
        targets = [b for b in blocks if any(prefix in b[4] for prefix in prefixes)]
        assert len(targets) == expected, (name, [b[4] for b in targets])
        for block in targets:
            page.add_redact_annot(pymupdf.Rect(block[:4]) + (-.3, -.3, .3, .3), fill=(1, 1, 1))
        page.apply_redactions(images=0, graphics=0, text=0)
        assert not any(prefix in page.get_text() for prefix in prefixes)
        stem = OUT / f'Fig{number}_comments_v5'
        doc.save(stem.with_suffix('.pdf'), garbage=4, deflate=True)
        page.get_pixmap(matrix=pymupdf.Matrix(600/72, 600/72), alpha=False).save(stem.with_suffix('.png'))
        doc.close()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before_hash
        records.append(dict(figure=number, source=str(source), source_sha256=before_hash,
                            removed_text=[b[4].strip() for b in targets],
                            images_and_vector_graphics_preserved=True, uncertainty_widths_unchanged=True))
        print(f'Fig. {number}: removed {len(targets)} vector-text production notes.', flush=True)
    (OUT / 'production_note_removal.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
