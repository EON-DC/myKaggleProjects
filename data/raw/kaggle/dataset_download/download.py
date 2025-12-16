import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# api.dataset_list_cli(sort_by='downloadCount')