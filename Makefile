.PHONY: clean preprocess train app

REMOTE := FALSE

DATA_DIR := dataset/ogbl_twitter/
INTER_DIR := $(addprefix $(DATA_DIR), intermediate/)
MAPPING_DIR := $(addprefix $(DATA_DIR), mapping/)
PROC_DIR := $(addprefix $(DATA_DIR), processed/)
SPLIT_DIR := $(addprefix $(DATA_DIR), split/target/)
DB_DIR := $(addprefix $(DATA_DIR), db/)


clean:
	rm -rf $(addprefix $(INTER_DIR), *)
	rm -rf $(addprefix $(MAPPING_DIR), *)
	rm -rf $(addprefix $(PROC_DIR), *)
	rm -rf $(addprefix $(SPLIT_DIR), *)
	rm -rf $(addprefix $(DB_DIR), *)


preprocess: 
	python src/synthetic_graph.py --num_partitions 5 --user_based --centralized --center_skew 1.0 --scenario threefold \


simulate:
	python src/main.py \
		--confirmation_bias 0.1 --recommendation_mode recommendation --condition conditional_ideological

	python src/main.py \
		--confirmation_bias 0.2 --recommendation_mode recommendation --condition conditional_ideological

	python src/main.py \
		--confirmation_bias 0.3 --recommendation_mode recommendation --condition conditional_ideological

	python src/main.py \
		--confirmation_bias 0.4 --recommendation_mode recommendation --condition conditional_ideological

	python src/main.py \
		--confirmation_bias 0.5 --recommendation_mode recommendation --condition conditional_ideological

	python src/main.py \
		--confirmation_bias 0.6 --recommendation_mode recommendation --condition conditional_ideological

	python src/main.py \
		--confirmation_bias 0.7 --recommendation_mode recommendation --condition conditional_ideological

	python src/main.py \
		--confirmation_bias 0.8 --recommendation_mode recommendation --condition conditional_ideological

	python src/main.py \
		--confirmation_bias 0.9 --recommendation_mode recommendation --condition conditional_ideological
		
full_run:
	python src/main.py \
  		--dataset synth_2_comms_gate09 --condition ideological --rank_users --latitude_of_acceptance 0.5 --cuda

	python src/main.py \
  		--dataset synth_2_comms_gate09 --condition epistemic --rank_users --latitude_of_acceptance 0.5 --cuda

	python src/main.py \
  		--dataset synth_2_comms_gate09 --condition ideological --latitude_of_acceptance 0.5 --cuda

	python src/main.py \
  		--dataset synth_2_comms_gate09 --condition epistemic --latitude_of_acceptance 0.5 --cuda
run:
	python src/main.py \
		--confirmation_bias 0.6 --recommendation_mode recommendation --condition epistemic


Purple: 5c3c92
Yellow: eed102