from pilot900_inputs import make_input
from adaptive_temporal_sampling import ROOT,read
def records():return read(ROOT/"artifacts/common/spatial_sampling_sources_v2/records.json")
