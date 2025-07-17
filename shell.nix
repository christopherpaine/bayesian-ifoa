{ pkgs ? import <nixpkgs> {} }:

let
  pythonEnv = pkgs.python3.withPackages (ps: with ps; [
#    pip

livereload
matplotlib
    tabulate
    numpy
    notebook
    pandas
#   dtale
#    qgrid
    jupyter
    jupytext
#    ipywidgets
    jupyterlab-git
    seaborn
    plotly
    openpyxl
    ipython
    # Add other packages as needed
  ]);
in
pkgs.mkShell {
  buildInputs = [ pythonEnv
    pkgs.nodejs
    pkgs.nodePackages.live-server





  ];
}

