


def balance_dataset(x_train, y_train):
    indices_0 = np.where(y_train == 0)[0] # MAJORITY CLASS
    indices_1 = np.where(y_train == 1)[0] # MINORITY CLASS

    count_0 = len(indices_0)
    count_1 = len(indices_1)

    target_count = (count_0 + count_1) // 2  # find a middle ground to avoid extremely oversampling the minority class
    #target_count = max(count_0, count_1)
    #target_count = min(count_0, count_1)

    oversampled_indices = np.random.choice(indices_1, size=target_count - count_1, replace=True)
    oversampled_X = x_train[oversampled_indices]
    oversampled_y = y_train[oversampled_indices]

    undersampled_indices = np.random.choice(indices_0, size=target_count - count_0, replace=False)
    undersampled_X = x_train[undersampled_indices]
    undersampled_y = y_train[undersampled_indices]

    balanced_X = np.vstack((undersampled_X, oversampled_X))
    balanced_y = np.hstack((undersampled_y, oversampled_y))

    balanced_data = np.column_stack((balanced_X, balanced_y))
    np.random.shuffle(balanced_data)
    shuffled_balanced_X = balanced_data[:, :-1]
    shuffled_balanced_y = balanced_data[:, -1]

    return shuffled_balanced_X, shuffled_balanced_y