#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:34:32 2024

@author: kudu
"""

class Helper:
    
    def __init__(self):
        pass
        
        
        
        
        
    def lines_splitter(self, ys):
        for index in range(len(ys)):
            if abs( ys[index] - ys[index + 1] ) > 20:
                line1 = ys[:index+1]
                line2 = ys[index+1:]
                break
        return line1, line2
    
    
    
    # Function to reorder chars
    # from the top to the bottom
    # from right to left
    def sort_contours(self, contours_rects):
        contours_rects = sorted(contours_rects, key=lambda x: x[0])
        ys             = [rect[1] for rect in contours_rects]
        ys             = sorted(list(set(ys)))[1:]
        
        line1, line2 = self.lines_splitter(ys)
        
        fstLine_cntrs = []
        sndLine_cntrs = []
        
        for record in contours_rects:
            if record[1] in line1:
                fstLine_cntrs.append(record[-1])
            elif record[1] in line2:
                sndLine_cntrs.append(record[-1])
            else:
                continue
        
        sorted_contours = fstLine_cntrs + sndLine_cntrs
        
        return sorted_contours