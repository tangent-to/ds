# Set up local environment

Data science with JavaScript is fun. Using it on the web on [Observable](https://observablehq.com/collection/@essi/data-science-with-tangent) is a charm. But using it locally will need you to open a terminal (or Powershell on Windows) and install/configure stuff. It mioght looks scary, but I'll guide you through it.

You need to install two engines: Deno for JavaScript and uv for Python. You need Python to install Jupyter, a superb environment to run code interactively - there are other ways to run JavaScript interactively, but I think Jupyter is the best as of today.

Following the oficial documentation, [install Deno](https://docs.deno.com/runtime/getting_started/installation/), then install [uv](https://docs.astral.sh/uv/#installation).

Once installed, you need to download jupyter and tell Deno to use it. In the terminal, install Jupyter lab (the "lab" is the modern version of Jupyter) with uv.

```bash
uv tool install jupyterlab
```

Then download the widget that allows Deno to use it.

```bash
deno jupyter --install
```

This will create a Jupyter kernel in your user account, which Jupyter is looking for to connect the interface (Jupyter) to the code interpreter (Deno). You might need to restart your terminal at this point. In any terminal, you can navigate between folders using the `cd` command (`c`hange `d`irectory), using `cd Documents` to got to the Documents floder and `cd ..` to go back one step. Type `ls` to see the files and folders of where you are. Once you end up in the folder of your project, type `jupyter lab` (don't forget the space between both), Jupyter wil open, hopefully with the Deno icon that allows you to code JavaScript in the notebook. 