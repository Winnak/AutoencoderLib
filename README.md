# Autoencoder lib

This is a wrapper around [Shark ML](http://www.shark-ml.org/) (4.0.0#[7a182c7923e94cf6a8d65b6c92a162bafad8314c](https://github.com/Shark-ML/Shark/tree/7a182c7923e94cf6a8d65b6c92a162bafad8314c)) for creating autoencoders.

While Shark does the fine (even has a [neat tutorial on it](http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/tutorials/algorithms/autoencoders.html))
I had a couple of issues integrating Shark into my master thesis project, due to dependencies in headers.

So I created this wrapper around an autoencoder configured for my thesis, made sure it compiled into a static lib, and that all dependencies are contained (so no boost in the header).

Hopefully this will make it pretty easy to integrate into projects

## Installation

1. Compile `AutoencoderLib.lib`
2. Link it to your own project alongside:
    * `libboost_filesystem`
    * `libboost_serialization`
    * `libboost_system`
3. include "Autoencoder.h"

I have not tested / compiled / used this on other platforms than Windows.

## License

Shark is released under GNU Lesser General Public License v3.0.

Therefore; this library is released under GNU Lesser General Public License v3.0.
