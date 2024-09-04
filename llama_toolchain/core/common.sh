# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

cleanup() {
  envname="$1"

  set +x
  echo "Cleaning up..."
  conda deactivate
  conda env remove --name $envname -y
}

handle_int() {
  if [ -n $ENVNAME ]; then
    cleanup $ENVNAME
  fi
  exit 1
}

handle_exit() {
  if [ $? -ne 0 ]; then
    echo -e "\033[1;31mABORTING.\033[0m"
    if [ -n $ENVNAME ]; then
      cleanup $ENVNAME
    fi
  fi
}

setup_cleanup_handlers() {
  trap handle_int INT
  trap handle_exit EXIT

  __conda_setup="$('conda' 'shell.bash' 'hook' 2>/dev/null)"
  eval "$__conda_setup"

  conda deactivate
}
