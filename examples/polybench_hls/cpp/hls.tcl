# Usage: vitis_hls -f hls.tcl -tclargs [impl] [top <name>] [hls_prj <project name>]

proc lshift listVar {
  upvar 1 $listVar l
  set r [lindex $l 0]
  set l [lreplace $l [set l 0] 0]
  return $r
}

set impl 0
set top kernel
set hls_prj benchmark.prj

while {[llength $argv]} {
  set flag [lshift argv]
  switch -exact -- $flag {
    impl {
      set impl 1
    }
    top {
      set top [lshift argv]
    }
    hls_prj {
      set hls_prj [lshift argv]
    }
  }
}

open_project ${hls_prj} -reset
set_top $top; # The name of the hardware function.
add_files [glob ./*.cpp] -cflags "-std=c++11 -DVHLS" ; # HLS source files.
open_solution "solution1"
set_part xcu250-figd2104-2-e
create_clock -period 7

# Actions we can take include csim_design, csynth_design, or cosim_design.
# csim_design
csynth_design
# cosim_design

if {$impl} {
  # Exporting the design gives us a convenient way to run place and route via
  # the `-flow impl` option. Another option is `-flow syn` if you only need to
  # run RTL synthesis.
  # The packaging options don't matter for our purposes, but we set the version
  # to 1.1.0 to circumvent this bug: https://support.xilinx.com/s/article/76960
  export_design -format ip_catalog -version 1.1.0 -flow impl
}

exit
