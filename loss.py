import tensorflow as tf


def cross_two_tasks_weight(y1, y2):
    p1, p2 = tf.keras.backend.sum(y1), tf.keras.backend.sum(y2)
    w1, w2 = p2 / (p1 + p2), p1 / (p1 + p2)
    return w1, w2


def balanced_entropy(x, y):
    eps = 1e-6
    z = tf.keras.activations.softmax(x)
    cliped_z = tf.keras.backend.clip(z, eps, 1 - eps)
    log_z = tf.keras.backend.log(cliped_z)

    num_classes = y.shape.as_list()[-1]
    ind = tf.keras.backend.argmax(y, axis=-1)
    total = tf.keras.backend.sum(y)

    m_c, n_c, loss = [], [], 0
    for c in range(num_classes):
        m_c.append(
            tf.keras.backend.cast(tf.keras.backend.equal(ind, c), dtype=tf.int32)
        )
        n_c.append(
            tf.keras.backend.cast(tf.keras.backend.sum(m_c[-1]), dtype=tf.float32)
        )

    c = []
    for i in range(num_classes):
        c.append(total - n_c[i])
    tc = tf.math.add_n(c)

    for i in range(num_classes):
        w = c[i] / tc
        m_c_one_hot = tf.one_hot((i * m_c[i]), num_classes, axis=-1)
        y_c = m_c_one_hot * y
        loss += w * tf.keras.backend.mean(-tf.keras.backend.sum(y_c * log_z, axis=1))

    return loss / num_classes
