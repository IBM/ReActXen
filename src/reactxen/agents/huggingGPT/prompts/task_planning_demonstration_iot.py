DEMONSTRATIONS = [
    {
        "Question": "What sites are there?",
        "Answer": '[{{"task": "sites", "id": 0, "dep": [-1], "args": {{}}}}]',
    },
    {
        "Question": "What assets are at site ABC?",
        "Answer": '[{{"task": "assets", "id": 0, "dep": [-1], "args": {{"site_name": "ABC"}}}}]',
    },
    {
        "Question": "Download asset metadata for Chiller 6 at XYZ site",
        "Answer": '[{{"task": "sensors", "id": 0, "dep": [-1], "args": {{"asset_name": "Chiller 6", "site_name": "XYZ"}}}}]'
    },
    {
        "Question": "Download asset metadata for Chiller 6 at XYZ site",
        "Answer": '[{{"task": "sensors", "id": 0, "dep": [-1], "args": {{"asset_name": "Chiller 6", "site_name": "XYZ"}}}}]'
    },
    {
        "Question": "Download asset history for sensor Chiller 3 Tonnage on asset Chiller 6 at XYZ site at 2015-09-19T23:45:00-04:00",
        "Answer": '[{{"task": "history", "id": 0, "dep": [-1], "args": {{"site_name": "XYZ", "asset_name_list": "Chiller 3", "sensor_name": "Chiller 3 Tonnage", "start": "2015-09-19T23:45:00-04:00", "final": null}}}}]'
    },
    {
        "Question": "Download all asset history for Chiller 6 at XYZ site from 2016-07-14T20:30:00-04:00 to 2016-07-14T23:30:00-04:00",
        "Answer": '[{{"task": "history", "id": 0, "dep": [-1], "args": {{"site_name": "XYZ", "asset_name_list": "Chiller 6", "start": "2016-07-14T20:30:00-04:00", "final": "2016-07-14T23:30:00-04:00"}}}}]'
    },
    {
        "Question": "Merge these JSON files file1.json and file2.json into a single JSON file",
        "Answer": '[{{"task": "jsonfilemerge", "id": 0, "dep": [-1], "args": {{"file_name_1": "file1.json", "file_name_2": "file2.json"}}}}]'
    },
    {
        "Question": "How do I learn the correct name for the sensor Condenser Return Temperature on Chiller 1 at site XYZ?",
        "Answer": '[{{"task": "sensors", "id": 0, "dep": [-1], "args": {{"asset_name": "Chiller 1", "site_name": "XYZ"}}}}]'
    },
]
