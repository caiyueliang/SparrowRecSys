import os
import argparse
import tensorflow as tf


# load sample as tf dataset
def get_dataset(file_path, batch_size=12):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=batch_size,
        label_name='label',
        # na_value="0",
        num_epochs=1,
        field_delim="\t",
        ignore_errors=True)
    return dataset


def DeepFM(drop_rate=0.2):
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
        'meat_type': tf.keras.layers.Input(name='meat_type', shape=(), dtype='string'),
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

    # ========================================================================================================
    # movie id embedding feature
    spu_vocab = ["农家小炒肉", "菠萝咕咾肉", "攸县香干炒肉", "回锅肉", "宫保鸡丁", "水煮肉片", "梅菜扣肉", "香辣紫苏牛蛙", "豆豉辣椒蒸排骨",
                 "酸菜鱼", "水煮鱼", "青椒肉丝", "麻婆豆腐", "小鸡炖蘑菇", "羊蝎子", "西芹百合炒腰果", "孜然牛肉", "干锅肥肠", "京酱肉丝",
                 "肉末酸豆角", "藜蒿炒腊肉", "酸辣土豆丝", "紫苏煎黄瓜", "清炒莴笋丝", "肉末煎豆腐", "糖醋排骨", "红烧猪蹄", "港式清汤腩",
                 "贝勒爷炒烤羊肉", "土匪猪肝", "糖醋里脊", "蟹黄豆腐", "本帮红烧肉", "芥蓝炒牛肉", "台式三杯鸡", "芥菜高汤煮肉片",
                 "湘西外婆菜", "香菜炒牛肉", "干锅花菜", "酸萝卜炒肚丝", "辣白菜炒五花肉", "猪肉炖粉条", "新疆大盘鸡", "毛血旺", "麻辣香锅",
                 "水煮牛肉", "小炒黄牛肉", "广式猪肚鸡", "韭黄炒鸡丝", "爆炒卤猪耳", "井冈烟笋炒腊肉", "辣炒鸡胗", "酸汤肥牛",
                 "五花肉炝炒包菜", "砂锅山药", "小炒双味菌", "鱼香肉丝", "鱼香豆腐", "红焖羊排", "酸菜白肉", "毛氏红烧肉", "蒜苔炒肉丝",
                 "茭白炒牛肉丝", "剁椒青豆肉末", "枸杞叶猪肝汤", "豆豉鲮鱼油麦菜", "番茄牛腩", "山药龙骨汤"]
    spu_col = tf.feature_column.categorical_column_with_vocabulary_list(key="spu_name",
                                                                        vocabulary_list=spu_vocab)
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
    cuisines_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="cuisines", vocabulary_list=cuisines_vocab)
    cuisines_genre_emb_col = tf.feature_column.embedding_column(cuisines_genre_col, 10)
    cuisines_genre_ind_col = tf.feature_column.indicator_column(cuisines_genre_col)

    meat_type_vocab = ["beef", "mutton", "fish", "pig", "chicken", "frog", "crab", "shrimp"]
    meat_type_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="meat_type", vocabulary_list=meat_type_vocab)
    meat_type_genre_emb_col = tf.feature_column.embedding_column(meat_type_genre_col, 10)
    meat_type_genre_ind_col = tf.feature_column.indicator_column(meat_type_genre_col)

    gender_genre_col = tf.feature_column.categorical_column_with_identity(key="user_gender", num_buckets=10)
    gender_genre_ind_col = tf.feature_column.indicator_column(gender_genre_col)

    # ========================================================================================================
    # The first-order term in the FM layer
    fm_first_order_columns = [spu_ind_col,
                              user_ind_col,
                              cuisines_genre_ind_col,
                              gender_genre_ind_col,
                              meat_type_genre_ind_col]
    fm_first_order_layer = tf.keras.layers.DenseFeatures(fm_first_order_columns)(inputs)
    fm_first_order_layer = tf.keras.layers.Dropout(drop_rate)(fm_first_order_layer)

    # ========================================================================================================
    spu_emb_layer = tf.keras.layers.DenseFeatures([spu_emb_col])(inputs)
    user_emb_layer = tf.keras.layers.DenseFeatures([user_emb_col])(inputs)
    cuisines_genre_emb_layer = tf.keras.layers.DenseFeatures([cuisines_genre_emb_col])(inputs)
    meat_type_genre_emb_layer = tf.keras.layers.DenseFeatures([meat_type_genre_emb_col])(inputs)
    gender_genre_emb_layer = tf.keras.layers.DenseFeatures([gender_genre_ind_col])(inputs)

    # FM part, cross different categorical feature embeddings
    product_layer_spu_user = tf.keras.layers.Dot(axes=1)([spu_emb_layer, user_emb_layer])
    product_layer_spu_gender_genre = tf.keras.layers.Dot(axes=1)([spu_emb_layer, gender_genre_emb_layer])
    product_layer_user_cuisines_genre = tf.keras.layers.Dot(axes=1)([user_emb_layer, cuisines_genre_emb_layer])
    product_layer_user_meat_type_genre = tf.keras.layers.Dot(axes=1)([user_emb_layer, meat_type_genre_emb_layer])
    product_layer_cuisines_genre_gender_genre = tf.keras.layers.Dot(axes=1)([cuisines_genre_emb_layer, gender_genre_emb_layer])
    product_layer_meat_type_genre_gender_genre = tf.keras.layers.Dot(axes=1)([meat_type_genre_emb_layer, gender_genre_emb_layer])

    product_layer_spu_user = tf.keras.layers.Dropout(drop_rate)(product_layer_spu_user)
    product_layer_spu_gender_genre = tf.keras.layers.Dropout(drop_rate)(product_layer_spu_gender_genre)
    product_layer_user_cuisines_genre = tf.keras.layers.Dropout(drop_rate)(product_layer_user_cuisines_genre)
    product_layer_user_meat_type_genre = tf.keras.layers.Dropout(drop_rate)(product_layer_user_meat_type_genre)
    product_layer_cuisines_genre_gender_genre = tf.keras.layers.Dropout(drop_rate)(product_layer_cuisines_genre_gender_genre)
    product_layer_meat_type_genre_gender_genre = tf.keras.layers.Dropout(drop_rate)(product_layer_meat_type_genre_gender_genre)

    # ========================================================================================================
    # deep part, MLP to generalize all input features

    deep_feature_columns = [tf.feature_column.numeric_column('sour'),
                            tf.feature_column.numeric_column('sweet'),
                            tf.feature_column.numeric_column('salty'),
                            tf.feature_column.numeric_column('hot'),
                            spu_emb_col,
                            user_emb_col]

    deep = tf.keras.layers.DenseFeatures(deep_feature_columns)(inputs)
    deep = tf.keras.layers.Dropout(drop_rate)(deep)
    deep = tf.keras.layers.Dense(64, activation='relu')(deep)
    deep = tf.keras.layers.Dropout(drop_rate)(deep)
    deep = tf.keras.layers.Dense(64, activation='relu')(deep)
    deep = tf.keras.layers.Dropout(drop_rate)(deep)

    # ========================================================================================================
    # concatenate fm part and deep part
    concat_layer = tf.keras.layers.concatenate(
        [fm_first_order_layer,
         product_layer_spu_user,
         product_layer_spu_gender_genre,
         product_layer_user_cuisines_genre,
         product_layer_user_meat_type_genre,
         product_layer_cuisines_genre_gender_genre,
         product_layer_meat_type_genre_gender_genre,
         deep], axis=1)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concat_layer)

    model = tf.keras.Model(inputs, output_layer)
    return model


def parse_argvs():
    parser = argparse.ArgumentParser(description='[DeepFM]')
    parser.add_argument("--train_data", type=str,
                        default="file:///data1/caiyueliang/data/user_goods_data_train_2022-05-05.csv")
    parser.add_argument("--test_data", type=str,
                        default="file:///data1/caiyueliang/data/user_goods_data_test_2022-05-05.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--drop_rate", type=float, default=0.2)

    args = parser.parse_args()
    print('[input params] {}'.format(args))

    return parser, args


if __name__ == "__main__":
    parser, args = parse_argvs()
    train_data = args.train_data
    test_data = args.test_data
    batch_size = args.batch_size
    epochs = args.epochs
    drop_rate = args.drop_rate

    # Training samples path, change to your local path
    training_samples_file_path = tf.keras.utils.get_file(train_data.split("/")[-1], train_data)
    # Test samples path, change to your local path
    test_samples_file_path = tf.keras.utils.get_file(test_data.split("/")[-1], test_data)

    # split as test dataset and training dataset
    train_dataset = get_dataset(training_samples_file_path, batch_size=batch_size)
    test_dataset = get_dataset(test_samples_file_path, batch_size=batch_size)

    model = DeepFM(drop_rate=drop_rate)

    # compile the model, set loss function, optimizer and evaluation metrics
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

    # train the model

    for i in range(epochs):
        model.fit(train_dataset, epochs=1)

        # evaluate the model
        test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
        print('\nTest Loss {:.4}, Test Accuracy {:.4}, Test ROC AUC {:.4}, Test PR AUC {:.4}'.format(
            test_loss, test_accuracy, test_roc_auc, test_pr_auc))

    # print some predict results
    predictions = model.predict(test_dataset)
    for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
        print("Predicted good rating: {:.2%}".format(prediction[0]),
              " | Actual rating label: ",
              ("Good Rating" if bool(goodRating) else "Bad Rating"))
