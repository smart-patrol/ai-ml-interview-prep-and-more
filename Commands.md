# VS Code Shortcuts

```
COMMAND LINE SHORTCUTS
	•	Esc + T: Swap the last two words before the cursor
	•	Ctrl + H: Delete the letter starting at the cursor
	•	Ctrl + W: Delete the word starting at the cursor
	•	TAB: Auto-complete files, directory, command names and much more
	•	Ctrl + R: To see the command history.
	•	Ctrl + U: Clear the line
	•	Ctrl + C: Cancel currently running commands.
	•	Ctrl + L: Clear the screen
	•	Ctrl + T: Swap the last two characters before the cursor
```
# Other Shorcuts

### Lock Screen
`Ctrl+cmd+q`

### Iterm DropDownTerminal
`Control + Space`


# START SESS

```
sudo apt-get install tmux

tmux new -s StreamSession
```

# FIND LARGE FILES

`find . -maxdepth 1 -printf '%s %p\n'|sort -nr|head`


# DOCKER

### nuke
`docker system prune -a`

### interactive
`docker run —rm -it -entrypoint bash <image-name-or-id>`

### NOT MAC1 M1 ARM
`docker build --platform=linux/amd64`

# TMux

```
tmux new -s StreamSession

tmux attach -t StreamSession
```

# TAR

```
tar -zcvf file.tar.gz /path/to/dir/

tar -xvf archive.tar.gz 
```

# Git Stuff

### To revert/reset to previous
`git reset --soft HEAD~1`

### Remove Files
`git rm --cached <file>`
`git commit --amend`

### check file is not there
`git ls-files`

`Git add workflow`

### Zsh useful git

```
glols
glol

gst=git status
ll=git log --oneline --graph --decorate --all
```

### other

`git tag -a "v0.0.2" -m "Second release of webserver-cluster"`

```
Git status
Git add . 
Git diff
Git diff —cached
Git commit -m
```

# MISC

remember my git keys mac
`ssh-add --apple-use-keychain `

JS for Video at 2.5x
`document.querySelector('video').playbackRate = 2.5;`
