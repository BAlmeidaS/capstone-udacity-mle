clean-all-data:
	@rm -rf project/data
	@git checkout project/data
	@echo 'all data downloaded were deleted!'

download-content:
	@python3 project/download-content.py

