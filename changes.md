# Explains for the "customization" branch

## Why would we need this branch?

This branch is intend for the adapation of imitation learning paradigms/data based current implementation of ***MALib***. 

## What have changed compared to the main branch?

Add serialization/deserialzation methods for both ```OfflineDatasetServer``` and ```ParameterServer```. Some naive unittest cases are attached as well.

## Some changes in the logical hierarchy
* OfflineDataServer:
  * To support loading and sampling from the offline expert dataset, we introduce the design of ```ExternalDataset```, which can be configured in the dict ```dataset_config```(please refer to ```runner.py``` and  ```settings.py```). The original offline dataset server is now treated as an interface for interactions with the original internal dataset(literally, the "main" dataset) and possibly external dataset. In the future, the offline dataset may have a series of internal datasets(can also be viewed as shards of data storage for load balancing, distributing and etc) and a series of external datasets(can be external database, offline data and other external resources, either being read-only or writeable). The return for a sampling request is a list contains the results from all of datasets.
  > [Return of Sampling Request] = ['main', 'extern I',  'extern II', ...]