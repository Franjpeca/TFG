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


def make_nominal_viz_wrapper(
    viz_func, metric, dataset_id, execution_folder, max_models_per_plot
):
    def _viz_nominal(*metrics_jsons):
        return viz_func(
            metrics_jsons=list(metrics_jsons),
            metric=metric,
            dataset_id=dataset_id,
            execution_folder=execution_folder,
            metric_type="nominal",
            max_models_per_plot=max_models_per_plot,
        )

    _viz_nominal.__name__ = f"viz_nominal_{metric}_{dataset_id}"
    return _viz_nominal


def make_ordinal_viz_wrapper(
    viz_func, metric, dataset_id, execution_folder, max_models_per_plot
):
    def _viz_ordinal(*metrics_jsons):
        return viz_func(
            metrics_jsons=list(metrics_jsons),
            metric=metric,
            dataset_id=dataset_id,
            execution_folder=execution_folder,
            metric_type="ordinal",
            max_models_per_plot=max_models_per_plot,
        )

    _viz_ordinal.__name__ = f"viz_ordinal_{metric}_{dataset_id}"
    return _viz_ordinal

