def make_predict_wrapper(predict_func, model_id, dataset_id):
    def _predict(model, dataset):
        return predict_func(model, dataset, model_id=model_id, dataset_id=dataset_id)
    _predict.__name__ = f"predict_{model_id}"
    return _predict

def make_evaluate_wrapper(evaluate_func, model_id, model_type, dataset_id):
    def _evaluate(model, dataset, y_pred, execution_folder):
        return evaluate_func(
            model=model,
            dataset=dataset,
            y_pred=y_pred,
            model_id=model_id,
            model_type=model_type,
            dataset_id=dataset_id,
            execution_folder=execution_folder,
        )
    _evaluate.__name__ = f"evaluate_{model_id}"
    return _evaluate