# Kara

Kara (**K**olouring **Ar**tificial **A**ssitant), is an AI bot the colourize B&W photos.

All the images used in the dataset are from [Unplash](https://unsplash.com/) and are creative common pictures by professional photographers. It includes 9.5 thousand training images and 500 validation images.

## Dependencies

- [Python 3.7](https://www.python.org/downloads/release/python-375/)

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
pip install tensorflow==2.0.0
```

Due to some error while locking packages stage, as can be seen in this [issue](https://github.com/pypa/pipenv/issues/3952), the tensorflow installation has to be manual.

**Observaion**: Again, if necessary, add the flag `--user` to make the pipenv package installation for the local user.

## References

Iizuka, Satoshi & Simo-Serra, Edgar & Ishikawa, Hiroshi. (2016). Let there be color!: joint end-to-end learning of global and local image priors for automatic image colorization with simultaneous classification. ACM Transactions on Graphics. 35. 1-11. 10.1145/2897824.2925974.

Wallner, Emil (Oct, 2017). Colorizing B&W Photos with Neural Networks. Floydhub. Access date: 7 Nov 2019. Available at: <https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/>
