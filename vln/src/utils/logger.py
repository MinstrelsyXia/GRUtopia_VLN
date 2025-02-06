#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging


class MyLogger(logging.Logger):
    def __init__(
        self,
        name,
        level,
        filename=None,
        filemode="a",
        stream=None,
        format_str=None,
        dateformat=None,
        style="%",
    ):
        super().__init__(name, level)
        self._formatter = logging.Formatter(format_str, dateformat, style)
        
        # 添加文件处理器
        if filename is not None:
            file_handler = logging.FileHandler(filename, filemode)  # type:ignore
            file_handler.setFormatter(self._formatter)
            super().addHandler(file_handler)
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler(stream)  # type:ignore
        console_handler.setFormatter(self._formatter)
        super().addHandler(console_handler)

    def add_filehandler(self, log_filename):
        """添加额外的文件处理器"""
        filehandler = logging.FileHandler(log_filename)
        filehandler.setFormatter(self._formatter)
        self.addHandler(filehandler)


logger = MyLogger(
    name="w61_grutopia", level=logging.INFO, format_str="%(asctime)-15s %(message)s"
)
