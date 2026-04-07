from presetdata import make_LS_dataset

if __name__ == '__main__':
    train_data, valid_data, test_data = make_LS_dataset()
    for batch_idx, batch_data in enumerate(train_data):
        print(f"Batch {batch_idx} 的完整对象:\n", batch_data)
        print("-" * 50)
        user_tensor = batch_data['user_id']  # [B]
        item_seq = batch_data['item_id_list']  # [B, MAX_ITEM_LIST_LENGTH]
        item_seq_len = batch_data['item_length']  # [B]
        target_item = batch_data['item_id']  # [B]

        print(batch_data['label_list'])
