from network.kara import Kara
import logging
import warnings
import sys

def execution_error(identifier):
    logging.error(f'[ERROR] Unknown error while executing KARA.'
                      f'\t\t\t\t\nTraceback: {identifier}')

def args_error():
    logging.error(f'[ERROR] No valid operation mode defined.\nRun: \
        \n\tpython3 main.py createmodel (to retrain the model) \
        \n\tpython3 main.py loadmodel (to load last trained model)'
    )

def load_model():
    try:
        assistant = Kara()
        assistant.assemble_from_file()
    except Exception as identifier:
        logging.error(f'[ERROR] Unknown error while executing KARA.'
                      f'\t\t\t\t\nTraceback: {identifier}')


def create_model():
    try:
        assistant = Kara()
        assistant.assemble(20)
    except Exception as identifier:
        execution_error(identifier)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        args_error()
    else:
        logging.basicConfig(format='%(asctime)s,%(msecs)-3d - %(name)-2s - '
                        '%(levelname)-2s => %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
        logging.info('[INFO] Creating KARA')
        warnings.simplefilter("ignore")

        if sys.argv[1] == 'createmodel':
            create_model()
        elif sys.argv[1] == 'loadmodel':
            load_model()
        else:
            args_error()

