from network.kara import Kara
import logging
import warnings


def main():
    logging.basicConfig(format='%(asctime)s,%(msecs)-3d - %(name)-2s - '
                        '%(levelname)-2s => %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    logging.info('[INFO] Creating KARA')
    warnings.simplefilter("ignore")
    try:
        assistant = Kara()
        assistant.assemble(20)
    except Exception as identifier:
        logging.error(f'[ERROR] Unknown error while executing KARA.'
                      f'\t\t\t\t\nTraceback: {identifier}')


if __name__ == '__main__':
    main()
