DEMONSTRATIONS = [
    {
        "role": "user",
        "content": "generate work order information for Condenser Performance KPI",  # noqa: E501
    },
    {
        "role": "assistant",
        "content": '[{{"task": "workorder_generation", "id": 0, "dep": [-1], "args": {{"kpi": "Condenser Performance"}}}}]',
    },
    {
        "role": "user",
        "content": "Generate guideline for analyzing asset health for closed-loop water-cooled chiller",  # noqa: E501
    },
    {
        "role": "assistant",
        "content": '[{{"task": "knowledge_extraction", "id": 0, "dep": [-1], "args": {{"knowledge_type" : "asset health", "asset_class": "closed-loop water-colled chiller"}}}}]',  # noqa: E501
    },
    #{
    #    "role": "user",
    #    "content": "Recommend failure mode and relevant sensor variables for chiller",  # noqa: E501
    #},
    #{
    #    "role": "assistant",
    #    "content": '[{{"task": "failure_mode_sensor_analytics", "id": 0, "dep": [-1], "args": {{"asset_class": "chiller"}}}}]',  # noqa: E501
    #},
    {
        "role": "user",
        "content": "Fetch metadata and data for Chiller 6 at POKMAIN site, do data quality check for both the tasks",
    },
    {
        "role": "assistant",
        "content": '[{{"task": "sensor_data_access", "id": 0, "dep": [-1], "args": {{"site": "POKMAIN", "assetName": "Chiller 6", "metaHistory": "metadata"}}}}, {{"task": "sensor_data_access", "id": 1, "dep": [-1], "args": {{"site": "POKMAIN", "assetName": "Chiller 6", "metaHistory": "data"}}}}, {{"task": "Data_Quality_Check_And_Processing", "id": 2, "dep": [0], "args": {{"dataset": "<resource-0>"}}, {{"task": "Data_Quality_Check_And_Processing", "id": 3, "dep": [1], "args": {{"dataset": "<resource-1>"}}}}]'
    },
    {
        "role": "user",
        "content": "Discover availability of sensors for Chiller 6 at POKMAIN",  # noqa: E501
    },
    {
        "role": "assistant",
        "content": '[{{"task": "sensor_data_access", "id": 0, "dep": [-1], "args": {{"site" : "POKMAIN", "assetName": "Chiller 6", "metaHistory": "metadata"}}}}]',  # noqa: E501
    },
    {
        "role": "user",
        "content": "Get all equipment instances of Chiller at POKMAIN",  # noqa: E501
    },
    {
        "role": "assistant",
        "content": '[{{"task": "sensor_data_access", "id": 0, "dep": [-1], "args": {{"site" : "POKMAIN", "assetName": "Chiller", "metaHistory": "instances"}}}}]',  # noqa: E501
    },
    {
        "role": "user",
        "content": "return Chiller 1 sensor data at POKMAIN site",
    },
    {
        "role": "assistant",
        "content": '[{{"task": "sensor_data_access", "id": 0, "dep": [-1], "args": {{"site" : "POKMAIN", "assetName": "Chiller 1", "metaHistory": "data"}}}}]',  # noqa: E501
    },
    {
        "role": "user",
        "content": "perform zero shot time series forecasting using two input files: a.txt as dataset path and b.txt as dataset config path",  # noqa: E501
    },
    {
        "role": "assistant",
        "content":'[{{"task": "tsfm_forecasting", "id": 1, "dep": [-1], "args": {{"dataset_path": "a.txt", "dataset_config_path": "b.txt", "model_source":"bam" , "model_type": "ttm"}}}}]'
    },
    {
        "role": "user",
        "content": "perform sensor relevancy using two input files: a.txt as failure mode discovery path and b.txt as asset metadata path",
    },
    {
        "role": "assistant",
        "content": '[{{"task": "sensor_relevancy", "id": 0, "dep": [-1], "args": {{"failure_mode_discovery_path": "a.txt", "asset_metadata_path": "b.txt"}}}}]'
    },
    {
        "role": "user",
        "content": "discover a failure mode of Chiller 9",
    },
    {
        "role": "assistant",
        "content": '[{{"task": "failure_mode_discovery", "id": 0, "dep": [-1], "args": {{"asset_class": "Chiller"}}}}]'
    },
    {
        "role": "user",
        "content": "Generate guideline for analyzing asset environmental health for closed-loop water-cooled chiller",  # noqa: E501
    },
    {
        "role": "assistant",
        "content": '[{{"task": "knowledge_extraction", "id": 0, "dep": [-1], "args": {{"knowledge_type" : "asset health", "asset_class": "closed-loop water-colled chiller"}}}}]',  # noqa: E501
    },
    # {
    #     "role": "user",
    #     "content": "Discover sensor association/relevancy regarding failure mode given the metadata of Chiller 1 at POKMAIN site"
    # },
    # {
    #     "role": "assistant",
    #     "content": 'fmsr stuff'
    # },
    #{
    #    "role": "user",
    #    "content": "Fetch metadata and data for Chiller 6 at POKMAIN site, discover a failure mode and sensor association/relevancy given the metadata of Chiller 6, and perform forecasting of relevant sensor for the failure mode",
    #},
    #{
    #    "role": "assistant",
    #    "content": '[{{"task": "sensor_data_access", "id": 0, "dep": [-1], "args": {{"site": "POKMAIN", "assetName": "Chiller 6", "metaHistory": "metadata"}}}}, {{"task": "failure_mode_discovery", "id": 1, "dep": [-1], "args": {{"asset_class": "Chiller"}}}}, {{"task": "sensor_relevancy", "id": 2, "dep": [0, 1], "args": {{"failure_mode_discovery_path": "<resource-1>", "asset_metadata_path": "<resource-0>"}}}}, {{"task": "sensor_data_access", "id": 3, "dep": [-1], "args": {{"site": "POKMAIN", "assetName": "Chiller 6", "metaHistory": "data"}}}}, {{"task": "tsfm_forecasting", "id": 4, "dep": [2, 3], "args": {{"dataset_path": "<resource-3>", "dataset_config_path": "<resource-2>", "model_type": "ttm", "model_source": "bam"}}}}]'
    #},
]
