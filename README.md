# Running geocoding on Climate Policy Radar data

This code uses [Refined](https://github.com/amazon-science/ReFinED) to perform fine-grained entity detection and linking to Wikipedia on the Climate Policy Radar dataset.

Using Wikipedia as gives you the option to use Wikidata to geocode non-human-settlement entities (such as forests, lakes, rivers etc). But, if you just want to geocode cities and regions, jump straight to [Using spaCy + geopy to extract only human settlements](#using-spacy--geopy-to-extract-only-human-settlements-eg-cities-regions)

## Prerequisites

This repo uses `poetry` for dependency management. Run:

- `poetry install`
- `poetry run python find_entities_refined.py --help`

This project uses our open data. Run `git clone https://huggingface.co/datasets/ClimatePolicyRadar/all-document-text-data` and use the local directory it's cloned to to interact with this code.

## Running the parser

RefinED is quite an old library (but the open source community has not been blessed with a better alternative since, as far as we know). This means:

- it's best set up *in its own environment* due to out-of-date dependencies of common libraries like pandas and transformers
- it's not compatible with `mps` devices. The inference device defaults to CUDA and falls back to CPU.

### How to use the outputs

It will produce one output JSONL file per document, lines of which will look a bit like this. These contain entity names, wikipedia and wikidata IDs and types (both coarse and fine-grained) – but **not geolocations**.

``` jsonl
{"id":"UNFCCC.document.i00001062.n0000_-3063318147919845615","text":"Figure 112 Spatial distribution of seasonal Rx 20 mm trend projection over Ethiopia under RCP 2.6 scenario from 2020 to 2050 for a) summer b) spring C) autumn and maximum CDD d) Summer e) spring f) autumn","spans":[{"text":"Ethiopia","start_index":75,"end_index":83,"type":"GPE","fine_grained_type":"sovereign state","id":"Q115","probability":0.9897,"wikipedia_title":"Ethiopia","wikidata_id":"Q115"}],"metadata":{"document_id":"UNFCCC.document.i00001062.n0000"}}
{"id":"UNFCCC.document.i00001062.n0000_3328379240098379050","text":"a) sdii(JJA)","spans":[{"text":"JJA","start_index":8,"end_index":11,"type":"unknown","fine_grained_type":"group of humans","id":"Q1418273","probability":0.9074,"wikipedia_title":"Jazz Journalists Association","wikidata_id":"Q1418273"}],"metadata":{"document_id":"UNFCCC.document.i00001062.n0000"}}
```

To filter on places, there are two options:

- Use the coarse type field: `type`. Filtering this on `type == "GPE"` will give you human settlements. But if you want to do this, you're better off using spaCy and geopy (see the next section).
- Use the `fine_grained_type` field. The taxonomy for this is assembled from the Wikidata concept hierarchies. There's no exhaustive list of the values of this anywhere, so you'll have to look at all of the outputs and filter to ones you find useful.

To fetch geolocations you can then use the Wikidata SPARQL endpoint as demonstrated in `fetch_coordinates_from_wikidata.py`. If there ends up being a very large number of entities you want to find locations for, it might be easier to download a [Wikidata data dump](https://www.wikidata.org/wiki/Wikidata:Database_download).

## Using spaCy + geopy to extract only human settlements (e.g. cities, regions)

If you just want to grab locations of cities and regions, you could replace the Refined logic with spaCy plus a [geopy](https://geopy.readthedocs.io/en/stable/#specifying-parameters-once) interfacing with your preferred service. There's a good guide to some services in their docs.

Here's an example of spaCy + geopy that Claude suggested if you're not familiar with the libraries

``` python
import spacy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Initialize the geocoder
geolocator = Nominatim(user_agent="my_agent")

def extract_locations(text):
    # Extract locations from text using spaCy
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "GPE"]

def geocode_location(location):
    # Geocode a location using GeoPy
    try:
        return geolocator.geocode(location)
    except (GeocoderTimedOut, GeocoderServiceError):
        time.sleep(1)  # Wait for a second before retrying
        return geocode_location(location)  # Retry

# Example text
text = "New York City is the most populous city in the United States."

# Extract locations
locations = extract_locations(text)

print("Extracted locations:")
for location in locations:
    print(f"- {location}")

print("\nGeocoding results:")
for location in locations:
    result = geocode_location(location)
    if result:
        print(f"{location}: {result.latitude}, {result.longitude}")
    else:
        print(f"{location}: Not found")
    time.sleep(1)  # Be nice to the geocoding service
```
