# Kara

[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org)

Kara (**K**olouring **Ar**tificial **A**ssitant), is an AI bot the colourize B&W photos.

![Kara](https://media.giphy.com/media/10mKMd68PI9jeU/giphy.gif)

All the images used in the dataset are from [Unplash](https://unsplash.com/) and are creative common pictures by professional photographers. It includes 9.5 thousand training images and 500 validation images.

## Usage

To retrain the model with the already available dataset, run:

```shell
cd Kara/
python3 main.py createmodel
```

To load the already trained model (there already is a trained model in the project):

```shell
cd Kara/
python3 main.py loadmodel
```

To predict specific images located in your filesystem, run:

```shell
cd Kara/
python3 main.py f <filename1> <filename2>
```

Each `<filename1>` represents a path to a specific image in your system. Multiple images are allowed, just separate each path with spaces.

## Dependencies

- [Python 3.7](https://www.python.org/downloads/release/python-375/)
- [Standard Version](https://github.com/conventional-changelog/standard-version)

## Configuration

To be done.

## Development

### Installing VirtualEnvWrapper

We recommend using a virtual environment created by the __virtualenvwrapper__ module. There is a virtual site with English instructions for installation that can be accessed [here](https://virtualenvwrapper.readthedocs.io/en/latest/install.html). But you can also follow these steps below for installing the environment:

```shell
sudo python3 -m pip install -U pip             # Update pip
sudo python3 -m pip install virtualenvwrapper  # Install virtualenvwrapper module
```

**Observation**: If you do not have administrator access on the machine remove `sudo` from the beginning of the command and add the flag `--user` to the end of the command.

Now configure your shell to use **virtualenvwrapper** by adding these two lines to your shell initialization file (e.g. `.bashrc`, `.profile`, etc.)

```shell
export WORKON_HOME=\$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
```

If you want to add a specific project location (will automatically go to the project folder when the virtual environment is activated) just add a third line with the following `export`:

```shell
export PROJECT_HOME=/path/to/project
```

Run the shell startup file for the changes to take effect, for example:

```shell
source ~/.bashrc
```

Now create a virtual environment with the following command (entering the name you want for the environment), in this example I will use the name **kara**:

```shell
mkvirtualenv -p $(which python3) kara
```

To use it:

```shell
workon kara
sudo python3 -m pip install pipenv
pipenv install # Will install all of the project dependencies
```

Install `tensorflow`:

```shell
pip install tensorflow==1.14
```

Due to some error while locking packages stage, as can be seen in this [issue](https://github.com/pypa/pipenv/issues/3952), the tensorflow installation has to be manual.

**Observaion**: Again, if necessary, add the flag `--user` to make the pipenv package installation for the local user.

### Local Execution

For local system execution, run the following command in the project root folder (assuming virtualenv is already active):

```shell
cd kara
python src/main.py createmodel
```

This will train and run the system on your machine. This way you can test new implementations or new optmizations. Also you can color the images for testing.

### Test

#### Lint

To lint your code follow the script bellow:

1. Enable virtualenv _color_;

2. Ensure that the dependencies are installed, especially:

```code
flake8
```

3. Run the command below:

```shell
cd kara/
flake8 src/
```

During the lint process the terminal will report a code errors and warnings from the PEP8 style guide, for more configurations and additional documentation go to [flake8](http://flake8.pycqa.org/en/latest/) and [PEP8](https://www.python.org/dev/peps/pep-0008/).

## Build

### Generate Changelog

To generate changelog we use the `standard version` tool, it will auto generate a new changelog for every new release by using the commit messages. To generate a new release and generate the updated changelog just do the following steps:

1. Install all dependencies

```shell
yarn install
```

2. Run standard version:

```shell
npm run release
```

If the release is a pre-release you should add the `--prerelease` to the command:

```shell
npm run release -- --prerelease alpha
```

For further instructions or other options check the full documentation of `standard version` project in the [CLI Usage](https://github.com/conventional-changelog/standard-version#cli-usage) section.

## Contributors

Project contributors

| Name | Registration |
| --- | --- |
| João Pedro Sconetto | 14/0145940 |
| Victor Correia de Moura | 15/0150792 |

## References

Iizuka, Satoshi & Simo-Serra, Edgar & Ishikawa, Hiroshi. (2016). Let there be color!: joint end-to-end learning of global and local image priors for automatic image colorization with simultaneous classification. ACM Transactions on Graphics. 35. 1-11. 10.1145/2897824.2925974.

Wallner, Emil (Oct, 2017). Colorizing B&W Photos with Neural Networks. Floydhub. Access date: 7 Nov 2019. Available at: <https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/>
