#!/bin/sh  
#!/bin/bash
############################################################
# Help: Arguments and help command                         #
############################################################
Help()
{
   # Display Help
   echo "ANDC Pipeline MFA"
   echo
   echo "Syntax: $0 -r root_dir"
   echo
}

while getopts 'r:' opt; do
        case $opt in
                r) root_dir=${OPTARG} ;;
                *)
                        echo 'Error in command line parsing' >&2
                        exit 1
        esac
done




python step2_mfa_preprocess.py --root $root_dir


# cd $root_dir/Outputs
mfa align $root_dir/Outputs/mfa_corpus english_us_arpa english_us_arpa $root_dir/Outputs/mfa_align_rsl --overwrite True --beam 1000 --retry_beam 4000 --clean
# cd -
python step2_mfa_postprocess.py --root $root_dir
