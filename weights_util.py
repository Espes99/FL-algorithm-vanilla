from encryption import encrypt_vector, load_CKKSVector_from_buffer, decrypt_vector


def encrypt_model_weights(weights, ckks_context):
    encrypted_weights = []
    original_shape = []
    for w in weights:
        original_shape.append(w.shape)
        flattened = w.flatten()
        encrypted_weight = encrypt_vector(ckks_context, flattened)
        encrypted_weights.append(encrypted_weight)
    return encrypted_weights, original_shape


def decrypt_model_weights(encrypted_weights, original_shapes, ckks_context):
    decrypted_weights = []

    for w, shape in zip(encrypted_weights, original_shapes):
        w_ser = w.serialize()
        encrypted_weight = load_CKKSVector_from_buffer(ckks_context, w_ser)
        decrypted_weight = decrypt_vector(ckks_context, encrypted_weight)
        decrypted_weight = decrypted_weight.reshape(shape)
        decrypted_weights.append(decrypted_weight)

    return decrypted_weights
