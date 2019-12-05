import tensorflow as tf


def pgd(x, y, model, eps=0.3, k=40, a=0.01):
    """ Projected gradient descent (PGD) attack
    """
    x_adv = tf.identity(x)
    loss_fn = tf.nn.softmax_cross_entropy_with_logits

    for _ in range(k):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_adv)
            logits, _, _, _, _ = model(x_adv, y)
            num_classes = tf.shape(logits)[1]
            labels = tf.one_hot(y, num_classes)
            loss = loss_fn(labels=labels, logits=logits)
        dl_dx = tape.gradient(loss, x_adv)
        x_adv += a * tf.sign(dl_dx)
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
    
    print("Finished attack", flush=True)
    return x_adv