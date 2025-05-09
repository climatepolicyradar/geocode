"""Example script showing how to fetch coordinates from Wikidata."""

import requests
import time

def fetch_geolocations(qid_list, batch_size=100):
    """Fetch geolocations from Wikidata for a list of QIDs given a SPARQL endpoint."""
    qid_list = list(set(qid_list))
    
    base_url = "https://query.wikidata.org/sparql"
    results = {}

    for i in range(0, len(qid_list), batch_size):
        batch = qid_list[i:i+batch_size]
        
        query = """
        SELECT ?item ?lat ?lon WHERE {
          VALUES ?item { %s }
          ?item wdt:P625 ?coord.
          ?item p:P625 ?statement.
          ?statement psv:P625 ?coordinate_node.
          ?coordinate_node wikibase:geoLatitude ?lat.
          ?coordinate_node wikibase:geoLongitude ?lon.
        }
        """ % " ".join(f"wd:{qid}" for qid in batch)

        # FIXME: set a responsible user agent
        headers = {
            'User-Agent': 'YourBotName/1.0 (your@email.com)'
        }

        response = requests.get(base_url, params={'query': query, 'format': 'json'}, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            for item in data['results']['bindings']:
                qid = item['item']['value'].split('/')[-1]
                lat = float(item['lat']['value'])
                lon = float(item['lon']['value'])
                results[qid] = (lat, lon)
        else:
            print(f"Error fetching batch: {response.status_code}")

        # Be nice to the Wikidata servers
        time.sleep(1)

    return results

if __name__ == "__main__":
    qid_list = ["Q2", "Q5", "Q30", "Q142", "Q183"]
    geolocations = fetch_geolocations(qid_list)

    for qid, coords in geolocations.items():
        print(f"{qid}: {coords}")
