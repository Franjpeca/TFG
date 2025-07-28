def make_train_wrapper(train_func, model_id, dataset_id):
    def _train(dataset, params, cv_settings):
        return train_func(
            dataset=dataset,
            params=params,
            cv_settings=cv_settings,
            model_id=model_id,
            dataset_id=dataset_id
        )
    _train.__name__ = f"train_{model_id}"
    return _train

def make_predict_wrapper(predict_func, model_id, dataset_id):
    def _predict(model, dataset):
        return predict_func(model, dataset, model_id=model_id, dataset_id=dataset_id)
    _predict.__name__ = f"predict_{model_id}"
    return _predict

def make_evaluate_wrapper(evaluate_func, model_id, model_type, dataset_id):
    def _evaluate(y_true, y_pred, model_params, execution_folder):
        return evaluate_func(
            y_true=y_true,
            y_pred=y_pred,
            model_params=model_params,
            execution_folder=execution_folder,
            model_id=model_id,
            model_type=model_type,
            dataset_id=dataset_id,
        )
    _evaluate.__name__ = f"evaluate_{model_id}"
    return _evaluate