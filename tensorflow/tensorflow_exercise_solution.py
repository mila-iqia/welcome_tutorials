# Avoid polluting the default graph by using an alternate graph
with tf.Graph().as_default():
    tf.set_random_seed(1234)

    # Create two scalar variables, x and y, initialized at random.
    x = tf.get_variable(name='x', shape=[], dtype=tf.float32,
                        initializer=tf.random_normal_initializer())
    y = tf.get_variable(name='y', shape=[], dtype=tf.float32,
                        initializer=tf.random_normal_initializer())

    # Create a tensor z whose value represents the expression
    #     2(x - 2)^2 + 2(y + 3)^2
    z = 2 * (x - 2) ** 2 + 2 * (y + 3) ** 2
    
    # Compute the gradients of z with respect to x and y.
    dx, dy = tf.gradients(z, [x, y])
    
    # Create an assignment expression for x using the update rule
    #    x <- x - 0.1 * dz/dx
    # and do the same for y.
    x_update = tf.assign_sub(x, 0.1 * dx)
    y_update = tf.assign_sub(y, 0.1 * dy)
    
    with tf.Session() as session:
        # Run the global initializer op for x and y.
        session.run(tf.global_variables_initializer())
        
        for _ in range(10):
            # Run the update ops for x and y.
            session.run([x_update, y_update])
            
            # Retrieve the values for x, y, and z, and print them.
            x_val, y_val, z_val = session.run([x, y, z])
            print('x = {:4.2f}, y = {:4.2f}, z = {:4.2f}'.format(x_val, y_val, z_val))
