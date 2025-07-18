## Purpose
a repo for the IfoA data science bayeisan inference working party so the team can easiy view where I am currently at with research.

## background
Having produce principal component analysis on nominal interest rates in the workplace i thought it would be interesting to formalise the incorporation of expert judgements via the bayesian framework. 

## developer stack
the analysis is using a highly customised (and enjoyable to work with) developer stack.  
### operating system
all work is done withing a linux NixOs build using Hyprland window manager.  Nixos is a great tool for the job enabling quick testing of new libraries and ensuring reproducibility.  the hyprland build is a fork of jakoolits hyprlnd... if you use linux i strongly suggest a look.  it is a very cool (slick) setup and very enjoyable to work with.  within my main machine is a dedicated nixos qemu VM using kde.  this is where the bulk of workflow occurs.  the VM can be snapshot at anytime meaning that getting started after a break is a breeze.  i also have a windows 10 vm that gives access to the msoffice suite.  it is not really needed but occassionally i .to_excel() dataframes to investigate differences before formalising the analysis in .ipynb.  libreoffice is usually good enough for these purposes however.  files are saved on an old laptop that has been repurposed to work as a NAS box and i use tailscale to provide a private net for ease of access.
### editor_
The editor of choice is neovim.  More specifically it is a nix build of neovim using the well regarded kickstart programme.  i have made a few customisations re: keyboard mappings and plugins.  the full repo is a public fork on my github account. This is used to edit .py and .tex files for the jupyter notebook and pdf research paper respectively.  
### scripts
this is couched with a number of bash scripts to produce realtime conversions to ipynb, html, pdf etc.  html pages are served using a simple flask app.  there is also a small amount of api integration with chatgpt.  if anyone is interested in the workflow (and enjoys discussing such things) give me a shout.  
### final thoughts
i have not put the files up here yet mainly as it would detract from the analysis, which is the ultimate aim.  however as the workflow develops, and becomes more efficient, i may do so if i think it would be of wider interest.  either way happy to discuss and cross fertlise ideas in this respect.

## frequency of updates
I plan to keep this updated at the end of each day with any progress made.
