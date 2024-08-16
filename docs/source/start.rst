⚙️ Quickstart Guide
===================

You can use `AutoMLQuantILDetect` in different ways.
There already exist quite some classifiers and AutoML tools which can be used to estimate mutual information using the log-loss and the accuracy of the learned model.


Fit a Classifier to Estimate MI
-------------------------------

Fit a ClassficationMIEstimator on a synthetic dataset using a random forest, estimate mutual information using the log-loss and the accuracy of the learned model and compare it with the ground-truth mutual information.
You can find similar example code snippets in
**examples/**.

.. code-block:: python

    from sklearn.metrics import accuracy_score
    from autoqild.dataset_readers.synthetic_data_generator import SyntheticDatasetGenerator
    from autoqild.mi_estimators.mi_estimator_classification import ClassficationMIEstimator
    from autoqild.utilities.constants import LOG_LOSS_MI_ESTIMATION, MID_POINT_MI_ESTIMATION

    # Step 1: Generate a Synthetic Dataset
    random_state = 42
    n_classes = 3
    n_features = 5
    samples_per_class = 200
    flip_y = 0.10  # Small amount of noise

    dataset_generator = SyntheticDatasetGenerator(
        n_classes=n_classes,
        n_features=n_features,
        samples_per_class=samples_per_class,
        flip_y=flip_y,
        random_state=random_state
    )

    X, y = dataset_generator.generate_dataset()

    print(f"Generated dataset X shape: {X.shape}, y shape: {y.shape}")

    # Step 2: Estimate Mutual Information using ClassficationMIEstimator
    mi_estimator = ClassficationMIEstimator(n_classes=n_classes, n_features=n_features, random_state=random_state)

    # Fit the estimator on the synthetic dataset
    mi_estimator.fit(X, y)

    # Estimate MI using log-loss
    estimated_mi_log_loss = mi_estimator.estimate_mi(X, y, method=LOG_LOSS_MI_ESTIMATION)
    estimated_mi_mid_point = mi_estimator.estimate_mi(X, y, method=MID_POINT_MI_ESTIMATION)
    # Step 3: Calculate Accuracy of the Model
    y_pred = mi_estimator.predict(X)
    accuracy = accuracy_score(y, y_pred)

    # Step 4: Compare with Ground-Truth MI
    ground_truth_mi = dataset_generator.calculate_mi()

    # Summary of Results
    print("##############################################################")
    print(f"Ground-Truth MI: {ground_truth_mi}")
    print(f"Estimated MI (Log-Loss): {estimated_mi_log_loss}")
    print(f"Estimated MI (Mid-Point): {estimated_mi_mid_point}")
    print(f"Model Accuracy: {accuracy}")

    >> Generated dataset X shape: (600, 5), y shape: (600,)
    >> ##############################################################
    >> Ground-Truth MI: 1.1751928845077875
    >> Estimated MI (Log-Loss): 1.3193094645863748
    >> Estimated MI (Mid-Point): 1.584961043823006
    >> Model Accuracy: 1.0

