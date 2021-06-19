import pymongo
from malib.rpc.ExperimentManager.mongo_server import MongoVisualizer as V

c = pymongo.MongoClient()
db = c["expr"]
col_names = db.list_collection_names()

col_select_msg = (
    "Please select collections to operated on:\n"
    + "\n".join([f"{idx} - {col_names[idx]}" for idx in range(len(col_names))])
    + "\n"
)
order = int(input(col_select_msg))
v = V(expr_name=col_names[order])
print("Visualizer handler 'v' is ready!")
