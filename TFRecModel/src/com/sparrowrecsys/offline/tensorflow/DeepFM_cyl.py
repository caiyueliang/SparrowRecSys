import tensorflow as tf

# # Training samples path, change to your local path
# training_samples_file_path = tf.keras.utils.get_file("trainingSamples.csv",
#                                                      "file:///Users/qudian/qudian-ml/SparrowRecSys/src/main"
#                                                      "/resources/webroot/sampledata/trainingSamples.csv")
# # Test samples path, change to your local path
# test_samples_file_path = tf.keras.utils.get_file("testSamples.csv",
#                                                  "file:///Users/qudian/qudian-ml/SparrowRecSys/src/main"
#                                                  "/resources/webroot/sampledata/testSamples.csv")

# /Users/qudian/open/cai/train_data/user_goods_data_2022-05-05.xlsx
# /Users/qudian/open/cai/train_data/user_goods_data_2022-05-05.csv
# /Users/qudian/open/cai/train_data/user_goods_data_train_2022-05-05.csv
# /Users/qudian/open/cai/train_data/user_goods_data_test_2022-05-05.csv

# Training samples path, change to your local path
training_samples_file_path = tf.keras.utils.get_file(
    "user_goods_data_train_2022-05-05.csv",
    "file:///Users/qudian/open/cai/train_data/user_goods_data_train_2022-05-05.csv")

# Test samples path, change to your local path
test_samples_file_path = tf.keras.utils.get_file(
    "user_goods_data_test_2022-05-05.csv",
    "file:///Users/qudian/open/cai/train_data/user_goods_data_test_2022-05-05.csv")


# load sample as tf dataset
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        na_value="0",
        num_epochs=1,
        field_delim="\t",
        ignore_errors=True)
    return dataset


# split as test dataset and training dataset
train_dataset = get_dataset(training_samples_file_path)
test_dataset = get_dataset(test_samples_file_path)

print("[train_dataset] \n {}".format(train_dataset))

# define input for keras model
# user_id	sku_id	spu_id	spu_name
# dinner_time_type	food_cook_type	goods_type	sku_valid_days	sku_line_price	sku_sale_price	sku_point_cnt	sku_spec
# spu_category	group_addition_info	sour	sweet	salty	hot	cook_method	meat_type	meat_weight	vagetables_weight
# total_weight	food_type	is_soup	key_words	cuisines	sku_cnt_t	label	user_gender	user_status	register_type
# spr_scheme_id	spr_batch_id	invite_user_id	invite_type	register_scene	is_head_flag	register_source	ip_city_name
# mobile_city_name	register_os	register_date	reg_gap_days	reg_channel	sum_invite_user_cnt_his
# sum_invite_valid_deal_user_cnt_his	last_receive_city_name	last_receive_region_name	last_valid_order_create_date
# last_valid_order_gap_days	sum_valid_deal_ord_cnt_his	sum_valid_deal_general_ord_cnt_his	sum_valid_deal_pin_ord_cnt_his
# sum_valid_deal_zd_ord_cnt_his	sum_valid_deal_amt_7d	sum_valid_deal_amt_30d	sum_valid_deal_amt_his
# sum_valid_deal_sku_cnt_7d	sum_valid_deal_sku_cnt_30d	sum_valid_deal_sku_cnt_his	sum_valid_deal_days_7d
# sum_valid_deal_days_30d	sum_valid_deal_days_his	valid_deal_most_spu_name	receive_time_range_top	order_hour_top
# order_hour_top2	order_hour_top3	valid_deal_spu_ctgy_top	valid_deal_spu_ctgy_top2	valid_deal_spu_ctgy_top3
# sum_valid_deal_ord_cnt_7d	sum_valid_deal_ord_cnt_30d

inputs = {
    'user_id': tf.keras.layers.Input(name='user_id', shape=(), dtype='int32'),
    'spu_name': tf.keras.layers.Input(name='spu_name', shape=(), dtype='string'),

    # 商品特征
    # 'dinner_time_type': tf.keras.layers.Input(name='dinner_time_type', shape=(), dtype='int32'),
    # 'food_cook_type': tf.keras.layers.Input(name='food_cook_type', shape=(), dtype='int32'),
    # 'goods_type': tf.keras.layers.Input(name='goods_type', shape=(), dtype='int32'),
    # 'sku_valid_days': tf.keras.layers.Input(name='goods_type', shape=(), dtype='int32'),
    # 'movieAvgRating': tf.keras.layers.Input(name='movieAvgRating', shape=(), dtype='float32'),
    #
    # 'sku_line_price': tf.keras.layers.Input(name='sku_line_price', shape=(), dtype='float32'),
    # 'sku_sale_price': tf.keras.layers.Input(name='sku_sale_price', shape=(), dtype='int32'),
    # 'sku_point_cnt': tf.keras.layers.Input(name='sku_point_cnt', shape=(), dtype='float32'),
    # 'spu_category': tf.keras.layers.Input(name='spu_category', shape=(), dtype='int32'),
    # 'group_addition_info': tf.keras.layers.Input(name='group_addition_info', shape=(), dtype='int32'),

    'sour': tf.keras.layers.Input(name='sour', shape=(), dtype='int32'),
    'sweet': tf.keras.layers.Input(name='sweet', shape=(), dtype='int32'),
    'salty': tf.keras.layers.Input(name='salty', shape=(), dtype='int32'),
    'hot': tf.keras.layers.Input(name='hot', shape=(), dtype='int32'),
    # 'cook_method': tf.keras.layers.Input(name='cook_method', shape=(), dtype='string'),
    # 'meat_type': tf.keras.layers.Input(name='meat_type', shape=(), dtype='string'),
    # 'meat_weight': tf.keras.layers.Input(name='meat_weight', shape=(), dtype='int32'),
    # 'vagetables_weight': tf.keras.layers.Input(name='vagetables_weight', shape=(), dtype='int32'),
    # 'total_weight': tf.keras.layers.Input(name='total_weight', shape=(), dtype='int32'),
    # 'food_type': tf.keras.layers.Input(name='food_type', shape=(), dtype='int32'),
    # 'is_soup': tf.keras.layers.Input(name='is_soup', shape=(), dtype='int32'),
    'cuisines': tf.keras.layers.Input(name='cuisines', shape=(), dtype='string'),

    # 用户特征
    'user_gender': tf.keras.layers.Input(name='user_gender', shape=(), dtype='int32'),
    'user_status': tf.keras.layers.Input(name='user_status', shape=(), dtype='int32'),
    # 'register_type': tf.keras.layers.Input(name='register_type', shape=(), dtype='int32'),
    # 'spr_scheme_id': tf.keras.layers.Input(name='spr_scheme_id', shape=(), dtype='int32'),
    # 'spr_batch_id': tf.keras.layers.Input(name='spr_batch_id', shape=(), dtype='int32'),
    # 'invite_user_id': tf.keras.layers.Input(name='invite_user_id', shape=(), dtype='int32'),
    # 'invite_type': tf.keras.layers.Input(name='invite_type', shape=(), dtype='int32'),
    # 'register_scene': tf.keras.layers.Input(name='register_scene', shape=(), dtype='int32'),
    # 'is_head_flag': tf.keras.layers.Input(name='is_head_flag', shape=(), dtype='int32'),
    # 'register_source': tf.keras.layers.Input(name='register_source', shape=(), dtype='int32'),
    # 'ip_city_name': tf.keras.layers.Input(name='ip_city_name', shape=(), dtype='string'),
    # 'mobile_city_name': tf.keras.layers.Input(name='mobile_city_name', shape=(), dtype='string'),
    # 'register_os': tf.keras.layers.Input(name='register_os', shape=(), dtype='string'),
    # # 'register_date': tf.keras.layers.Input(name='register_date', shape=(), dtype='string'),
    # 'reg_gap_days': tf.keras.layers.Input(name='reg_gap_days', shape=(), dtype='int32'),
    # # 'reg_channel': tf.keras.layers.Input(name='reg_channel', shape=(), dtype='string'),
    # 'sum_invite_user_cnt_his': tf.keras.layers.Input(name='sum_invite_user_cnt_his', shape=(), dtype='int32'),
    # 'sum_invite_valid_deal_user_cnt_his': tf.keras.layers.Input(name='sum_invite_valid_deal_user_cnt_his', shape=(), dtype='int32'),
    # 'last_receive_city_name': tf.keras.layers.Input(name='last_receive_city_name', shape=(), dtype='string'),
    # 'last_receive_region_name': tf.keras.layers.Input(name='last_receive_region_name', shape=(), dtype='string'),
    # 'last_valid_order_create_date': tf.keras.layers.Input(name='last_valid_order_create_date', shape=(), dtype='string'),
    # 'last_valid_order_gap_days': tf.keras.layers.Input(name='last_valid_order_gap_days', shape=(), dtype='int32'),
    # 'sum_valid_deal_ord_cnt_his': tf.keras.layers.Input(name='sum_valid_deal_ord_cnt_his', shape=(), dtype='int32'),
    # 'sum_valid_deal_general_ord_cnt_his': tf.keras.layers.Input(name='sum_valid_deal_general_ord_cnt_his', shape=(), dtype='int32'),
    # 'sum_valid_deal_pin_ord_cnt_his': tf.keras.layers.Input(name='sum_valid_deal_pin_ord_cnt_his', shape=(), dtype='int32'),
    # 'sum_valid_deal_zd_ord_cnt_his': tf.keras.layers.Input(name='sum_valid_deal_zd_ord_cnt_his', shape=(), dtype='int32'),
    # 'sum_valid_deal_amt_7d': tf.keras.layers.Input(name='sum_valid_deal_amt_7d', shape=(), dtype='float32'),
    # 'sum_valid_deal_amt_30d': tf.keras.layers.Input(name='sum_valid_deal_amt_30d', shape=(), dtype='float32'),
    # 'sum_valid_deal_amt_his': tf.keras.layers.Input(name='sum_valid_deal_amt_his', shape=(), dtype='float32'),
    # 'sum_valid_deal_sku_cnt_7d': tf.keras.layers.Input(name='sum_valid_deal_sku_cnt_7d', shape=(), dtype='int32'),
    # 'sum_valid_deal_sku_cnt_30d': tf.keras.layers.Input(name='sum_valid_deal_sku_cnt_30d', shape=(), dtype='int32'),
    # 'sum_valid_deal_sku_cnt_his': tf.keras.layers.Input(name='sum_valid_deal_sku_cnt_his', shape=(), dtype='int32'),
    # 'sum_valid_deal_days_7d': tf.keras.layers.Input(name='sum_valid_deal_days_7d', shape=(), dtype='int32'),
    # 'sum_valid_deal_days_30d': tf.keras.layers.Input(name='sum_valid_deal_days_30d', shape=(), dtype='int32'),
    # 'sum_valid_deal_days_his': tf.keras.layers.Input(name='sum_valid_deal_days_his', shape=(), dtype='int32'),
    # 'valid_deal_most_spu_name': tf.keras.layers.Input(name='valid_deal_most_spu_name', shape=(), dtype='string'),
    # 'receive_time_range_top': tf.keras.layers.Input(name='receive_time_range_top', shape=(), dtype='string'),
    # 'order_hour_top': tf.keras.layers.Input(name='order_hour_top', shape=(), dtype='string'),
    # 'order_hour_top2': tf.keras.layers.Input(name='order_hour_top2', shape=(), dtype='string'),
    # 'order_hour_top3': tf.keras.layers.Input(name='order_hour_top3', shape=(), dtype='string'),
    # 'valid_deal_spu_ctgy_top': tf.keras.layers.Input(name='valid_deal_spu_ctgy_top', shape=(), dtype='string'),
    # 'valid_deal_spu_ctgy_top2': tf.keras.layers.Input(name='valid_deal_spu_ctgy_top2', shape=(), dtype='string'),
    # 'valid_deal_spu_ctgy_top3': tf.keras.layers.Input(name='valid_deal_spu_ctgy_top3', shape=(), dtype='string'),
    # 'sum_valid_deal_ord_cnt_7d': tf.keras.layers.Input(name='sum_valid_deal_ord_cnt_7d', shape=(), dtype='int32'),
    # 'sum_valid_deal_ord_cnt_30d': tf.keras.layers.Input(name='sum_valid_deal_ord_cnt_30d', shape=(), dtype='int32'),
}

# inputs = {
#     'movieAvgRating': tf.keras.layers.Input(name='movieAvgRating', shape=(), dtype='float32'),
#     'movieRatingStddev': tf.keras.layers.Input(name='movieRatingStddev', shape=(), dtype='float32'),
#     'movieRatingCount': tf.keras.layers.Input(name='movieRatingCount', shape=(), dtype='int32'),
#     'userAvgRating': tf.keras.layers.Input(name='userAvgRating', shape=(), dtype='float32'),
#     'userRatingStddev': tf.keras.layers.Input(name='userRatingStddev', shape=(), dtype='float32'),
#     'userRatingCount': tf.keras.layers.Input(name='userRatingCount', shape=(), dtype='int32'),
#     'releaseYear': tf.keras.layers.Input(name='releaseYear', shape=(), dtype='int32'),
#
#     'movieId': tf.keras.layers.Input(name='movieId', shape=(), dtype='int32'),
#     'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
#     'userRatedMovie1': tf.keras.layers.Input(name='userRatedMovie1', shape=(), dtype='int32'),
#
#     'userGenre1': tf.keras.layers.Input(name='userGenre1', shape=(), dtype='string'),
#     'userGenre2': tf.keras.layers.Input(name='userGenre2', shape=(), dtype='string'),
#     'userGenre3': tf.keras.layers.Input(name='userGenre3', shape=(), dtype='string'),
#     'userGenre4': tf.keras.layers.Input(name='userGenre4', shape=(), dtype='string'),
#     'userGenre5': tf.keras.layers.Input(name='userGenre5', shape=(), dtype='string'),
#
#     'movieGenre1': tf.keras.layers.Input(name='movieGenre1', shape=(), dtype='string'),
#     'movieGenre2': tf.keras.layers.Input(name='movieGenre2', shape=(), dtype='string'),
#     'movieGenre3': tf.keras.layers.Input(name='movieGenre3', shape=(), dtype='string'),
# }

# # movie id embedding feature
# movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=1001)
# movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)
# movie_ind_col = tf.feature_column.indicator_column(movie_col) # movid id indicator columns
#
# # user id embedding feature
# user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
# user_emb_col = tf.feature_column.embedding_column(user_col, 10)
# user_ind_col = tf.feature_column.indicator_column(user_col) # user id indicator columns

# movie id embedding feature
spu_col = tf.feature_column.categorical_column_with_identity(key='spu_name', num_buckets=100)
spu_emb_col = tf.feature_column.embedding_column(spu_col, 10)
spu_ind_col = tf.feature_column.indicator_column(spu_col)       # spu id indicator columns

# user id embedding feature
user_col = tf.feature_column.categorical_column_with_identity(key='user_id', num_buckets=312000)
user_emb_col = tf.feature_column.embedding_column(user_col, 10)
user_ind_col = tf.feature_column.indicator_column(user_col)         # user id indicator columns


# ========================================================================================================
# genre features vocabulary
# cuisines genre embedding feature
cuisines_vocab = ["湘菜", "川菜", "鲁菜", "浙菜", "粤菜", "闽菜", "苏菜", "徽菜", "沪菜", "东北菜", "西北菜"]
cuisines_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="cuisines",
                                                                               vocabulary_list=cuisines_vocab)
cuisines_genre_emb_col = tf.feature_column.embedding_column(cuisines_genre_col, 10)
cuisines_genre_ind_col = tf.feature_column.indicator_column(cuisines_genre_col) # user genre indicator columns

# item genre embedding feature
gender_genre_col = tf.feature_column.categorical_column_with_identity(key="user_gender", vocabulary_list=2)
gender_genre_emb_col = gender_genre_col
gender_genre_ind_col = tf.feature_column.indicator_column(gender_genre_col)     # item genre indicator columns

# ========================================================================================================
# fm first-order term columns: without embedding and concatenate to the output layer directly
# fm_first_order_columns = [movie_ind_col, user_ind_col, user_genre_ind_col, item_genre_ind_col]
fm_first_order_columns = [spu_col, user_ind_col, cuisines_genre_ind_col, gender_genre_ind_col]


spu_emb_layer = tf.keras.layers.DenseFeatures([spu_emb_col])(inputs)
user_emb_layer = tf.keras.layers.DenseFeatures([user_emb_col])(inputs)
cuisines_genre_emb_layer = tf.keras.layers.DenseFeatures([cuisines_genre_emb_col])(inputs)
gender_genre_emb_layer = tf.keras.layers.DenseFeatures([gender_genre_emb_col])(inputs)


# The first-order term in the FM layer
fm_first_order_layer = tf.keras.layers.DenseFeatures(fm_first_order_columns)(inputs)

# FM part, cross different categorical feature embeddings
product_layer_spu_user = tf.keras.layers.Dot(axes=1)([spu_emb_layer, user_emb_layer])
product_layer_cuisines_genre_gender_genre = tf.keras.layers.Dot(axes=1)([cuisines_genre_emb_layer, gender_genre_emb_layer])
product_layer_spu_gender_genre = tf.keras.layers.Dot(axes=1)([spu_emb_layer, gender_genre_emb_layer])
product_layer_user_cuisines_genre = tf.keras.layers.Dot(axes=1)([user_emb_layer, cuisines_genre_emb_layer])

# ========================================================================================================
# deep part, MLP to generalize all input features

deep_feature_columns = [tf.feature_column.numeric_column('sour'),
                        tf.feature_column.numeric_column('sweet'),
                        tf.feature_column.numeric_column('salty'),
                        tf.feature_column.numeric_column('hot'),
                        spu_emb_col,
                        user_emb_col]

deep = tf.keras.layers.DenseFeatures(deep_feature_columns)(inputs)
deep = tf.keras.layers.Dense(64, activation='relu')(deep)
deep = tf.keras.layers.Dense(64, activation='relu')(deep)

# ========================================================================================================
# concatenate fm part and deep part
concat_layer = tf.keras.layers.concatenate(
    [fm_first_order_layer, product_layer_spu_user, product_layer_cuisines_genre_gender_genre,
     product_layer_spu_gender_genre, product_layer_user_cuisines_genre, deep], axis=1)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concat_layer)

# ========================================================================================================
model = tf.keras.Model(inputs, output_layer)

# ========================================================================================================
# compile the model, set loss function, optimizer and evaluation metrics
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

# ========================================================================================================
# train the model
model.fit(train_dataset, epochs=5)

# ========================================================================================================
# evaluate the model
test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
                                                                                   test_roc_auc, test_pr_auc))
# ========================================================================================================
# print some predict results
predictions = model.predict(test_dataset)
for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print("Predicted good rating: {:.2%}".format(prediction[0]),
          " | Actual rating label: ",
          ("Good Rating" if bool(goodRating) else "Bad Rating"))


# model.save("./DeepFM.mdl")
# model.save_weights("./DeepFM.weights")
