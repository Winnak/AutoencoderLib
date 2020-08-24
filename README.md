# Autoencoder lib

This is a wrapper around [Shark ML](http://www.shark-ml.org/) for creating autoencoders.

While Shark does the fine (even has a [neat tutorial on it](http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/tutorials/algorithms/autoencoders.html))
I had a couple of issues integrating Shark into my master thesis project, because of the boost dependency.

So I created this wrapper around an autoencoder configured for my thesis, made sure it compiled into a static lib, and that all dependencies were internal (so no boost in the header).
Hopefully this will make it pretty easy to integrate into projects, since installation is: 1. link .lib, 2. include "Autoencoder.h".

## License

Shark is released under GNU Lesser General Public License v3.0.

Therefore; this library is released under GNU Lesser General Public License v3.0.
