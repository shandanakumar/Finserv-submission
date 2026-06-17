
import sys, os
sys.path.insert(0, '.')
from pdfminer.high_level import extract_text
text = extract_text('real_docs/RBI-Master Direction - Know Your Customer (KYC) Direction.pdf')
# Search for re-verification content
lines = text.split('\n')
for i, line in enumerate(lines):
    if any(k in line.lower() for k in ['periodic', 're-kyc', 'updation', 'high risk', '2 year', '8 year']):
        print(f'Line {i}: {line.strip()[:120]}')