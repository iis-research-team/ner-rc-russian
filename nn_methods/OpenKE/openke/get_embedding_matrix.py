import json
import numpy as py

import config
import models
con = config.Config()
con.set_in_path("../benchmarks/wikidata/")
con.set_test_flag(True)
con.set_work_threads(16)
con.set_dimension(100)
con.set_import_files("../checkpoints/transe.ckpt")
con.init()
con.set_model(models.TransE)
# Get the embeddings (numpy.array)
embeddings = con.get_parameters("numpy")
# Get the embeddings (python list)
embeddings = con.get_parameters()
print(embeddings.shape)
