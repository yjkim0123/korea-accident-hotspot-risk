"""
Collect Korea traffic-accident frequent-zone data (data.go.kr dataset 15057467,
한국도로교통공단 지자체별 교통사고 다발지역).

Endpoint: B552061/frequentzoneLg/getRestFrequentzoneLg

You need your own free data.go.kr service key, and the 활용신청 for dataset
15057467 must be activated (otherwise the API returns HTTP 403). Supply the key
via the environment:

    export DATA_GO_KR_KEY='your-service-key'
    python3 collect_accidents.py

Output: data/kaccident_hotspots.csv with columns
  year, sido, gugun, spot_name, lat, lon, accident_cnt, casualty_cnt, ...

The collected file is already committed to data/ so the modelling scripts can be
run without a key; re-run this only to refresh or extend the panel.
"""
import csv, json, time, ssl, urllib.request, urllib.parse, os, sys

KEY = os.environ.get('DATA_GO_KR_KEY')
if not KEY:
    sys.exit("Set DATA_GO_KR_KEY to your data.go.kr service key "
             "(see the module docstring).")

BASE = "http://apis.data.go.kr/B552061/frequentzoneLg/getRestFrequentzoneLg"
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE

# 17 시도 코드 (도로교통공단 코드 체계: siDo 4-digit, guGun 5-digit).
SIDO = ['1100', '2600', '2700', '2800', '2900', '3000', '3100', '3600',
        '4100', '4300', '4400', '4500', '4600', '4700', '4800', '5000', '5100']  # 서울~제주
YEARS = [str(y) for y in range(2015, 2023)]


def fetch(year, sido, gugun, page=1, rows=100):
    q = urllib.parse.urlencode({'serviceKey': KEY, 'searchYearCd': year, 'siDo': sido,
                                'guGun': gugun, 'type': 'json', 'numOfRows': rows, 'pageNo': page})
    r = urllib.request.urlopen(BASE + '?' + q, timeout=30, context=CTX)
    return json.loads(r.read().decode('utf-8', 'ignore'))


def main():
    os.makedirs('data', exist_ok=True)
    out = open('data/kaccident_hotspots.csv', 'w', newline='', encoding='utf-8-sig')
    w = csv.writer(out); header_written = False; total = 0; _cols = []
    for year in YEARS:
        for sido in SIDO:
            try:
                d = fetch(year, sido, '')  # guGun blank = whole sido
            except Exception as e:
                print('skip', year, sido, str(e)[:80]); continue
            items = (((d.get('items') or {}).get('item')) if isinstance(d.get('items'), dict)
                     else d.get('items')) or []
            if isinstance(items, dict):
                items = [items]
            for it in items:
                if not header_written:
                    _cols = list(it.keys()); w.writerow(['year', 'sido'] + _cols); header_written = True
                w.writerow([year, sido] + [it.get(c, '') for c in _cols]); total += 1
            time.sleep(0.2)
        print('year done', year, 'total', total)
    out.close(); print('SAVED data/kaccident_hotspots.csv rows', total)


if __name__ == '__main__':
    main()
