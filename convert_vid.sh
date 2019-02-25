#!/usr/bin/env bash
SRC_VID = $1
DST_VID = $2
# convert to sort_edge_length = 360
# ffmpeg -y -i ${SRC_VID} -c:v mpeg4 -filter:v "scale=min(iw\,(360*iw)/min(iw\,ih)):-1" -b:v 640k -an ${DST_VID}
# or, convert to sort_edge_length = 256
ffmpeg -y -i ${SRC_VID} -c:v mpeg4 -filter:v "scale=min(iw\,(256*iw)/min(iw\,ih)):-1" -b:v 512k -an ${DST_VID}
# or, convert to sort_edge_length = 160
# ffmpeg -y -i ${SRC_VID} -c:v mpeg4 -filter:v "scale=min(iw\,(160*iw)/min(iw\,ih)):-1" -b:v 240k -an ${DST_VID}