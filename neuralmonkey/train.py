"""
This is a training script for sequence to sequence learning.
"""
# tests: lint, mypy
def create_config(config_file):
    config = Configuration()
    config.add_argument('name', str)

    config.add_argument('threads', int, required=False, default=4)

    return config.load_file(config_file)

def main():
    sess, saver = initialize_tf(args.initial_variables, args.threads)
    training_loop(sess, saver, args.epochs, args.trainer,
                  args.encoders + [args.decoder], args.decoder,
                  args.batch_size, args.train_dataset, args.val_dataset,
                  args.output, args.evaluation, args.runner,
                  test_datasets=args.test_datasets,
                  save_n_best_vars=args.save_n_best,
                  link_best_vars=link_best_vars,
                  vars_prefix=variables_file_prefix,
                  logging_period=args.logging_period,
                  validation_period=args.validation_period,
                  postprocess=args.postprocess,
                  minimize_metric=args.minimize)
