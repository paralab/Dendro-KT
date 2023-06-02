#!/bin/bash

# Set environment variable SRC_DIR to the Dendro-KT root directory.

# Argument 1 is filename with metadata like
#
#     prefix=MyData
#     title=Eureka
#     dim=2
#     nfields=1
#     field_default=1
#     comm_size=8

# Argument 2 (optional) can be used to override a default field.
#
#     ./raw_nodal_binary.sh solution.meta 2
#

set -e

IFS="="
while read -r name value
do
  if [ "$name" == "prefix" ] ; then
    prefix="$value"
  fi

  if [ "$name" == "suffix" ] ; then
    suffix="$value"
  fi

  if [ "$name" == "title" ] ; then
    title="$value"
  fi

  if [ "$name" == "dim" ] ; then
    dim="$value"
  fi

  if [ "$name" == "nfields" ] ; then
    nfields="$value"
  fi

  if [ "$name" == "field_default" ] ; then
    field="$value"
  fi

  if [ "$name" == "comm_size" ] ; then
    comm_size="$value"
  fi
done < "$1"

if [ "$2" != "" ] ; then
  field="$2"
fi

echo "Reading..."
eval ls -1 "$prefix"{0..$(($comm_size - 1))}"$suffix"

{ eval cat "$prefix"{0..$(($comm_size - 1))}"$suffix" ; } | 3<&0 gnuplot -e "set title $title; dim=$dim; nfields=$nfields; field=$field;" "$SRC_DIR"/IO/gnuplot/nodal_binary.gnu

