import mlflow
import os
# set tracking server and azure blob token
os.environ['AZURE_STORAGE_ACCESS_KEY']= "unLKPr/7WwSIIdnayitKbZUDezslfAcvqtk3F6IxYkRgeoe9nP6KHxVtZ03aAcJEkp9GUx6zXnXoG/QV2EQ3oA=="

def trigger_mlflow_run(tracking_uri, uri, model, dataset,
                       model_uri_origin="Blob Storage",
                       entry_point="test",
                       gpu="0", 
                       experiment_name="test_inference",
                       version="mlflow-test"):

    # os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    # if uri.split(".")[-1] != "git":
    #     raise Exception("Git URI is not valid!")
    # run_info = mlflow.projects.run( uri=uri, 
    #                                 entry_point=entry_point,
    #                                 docker_args={'gpus':f'device={gpu}'},
    #                                 parameters={"model_uri":model,
    #                                             "dataset_uri":dataset,
    #                                             "model_uri_origin":model_uri_origin},
    #                                 experiment_name=experiment_name,
    #                                 version=version)
    # status = run_info.get_status()
    # return run_info.run_id, status
    return 11, "FINISHED"
    # import time
    # time.sleep(10)    