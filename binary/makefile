# These are the names of the targets
CLEAN_TARGET := clean
HELP_TARGET := help

# These are the flags and command for the clean target
DELETE_FLAGS :=
DELETE_CMD := rm

# This is the compiler and the compile flags you want to use
COMPILER := gcc
COMPILE_FLAGS := -Wall -Werror -g -Og -std=gnu99 -oFast
LINKER_FLAGS := -lm

SOURCE_DIR := ../source
OBJECT_DIR := ../object
BINARY_DIR := ../binary

# These are the source files and object files for the review library
REVIEW_DIR := $(SOURCE_DIR)/review

REVIEW_SOURCE_FILES := $(wildcard $(REVIEW_DIR)/*.c)
REVIEW_OBJECT_FILES := $(addprefix $(OBJECT_DIR)/, $(notdir $(REVIEW_SOURCE_FILES:.c=.o)))

# These are the source files and object files for the persue library
PERSUE_DIR := $(SOURCE_DIR)/persue

PERSUE_SOURCE_FILES := $(wildcard $(PERSUE_DIR)/*.c)
PERSUE_OBJECT_FILES := $(addprefix $(OBJECT_DIR)/, $(notdir $(PERSUE_SOURCE_FILES:.c=.o)))

# These are the source files and object files for the secure library
SECURE_DIR := $(SOURCE_DIR)/secure

SECURE_SOURCE_FILES := $(wildcard $(SECURE_DIR)/*.c)
SECURE_OBJECT_FILES := $(addprefix $(OBJECT_DIR)/, $(notdir $(SECURE_SOURCE_FILES:.c=.o)))

all: master

%: $(OBJECT_DIR)/%.o $(SOURCE_DIR)/%.c $(REVIEW_OBJECT_FILES) $(REVIEW_SOURCE_FILES) $(PERSUE_OBJECT_FILES) $(PERSUE_SOURCE_FILES) $(SECURE_OBJECT_FILES) $(SECURE_SOURCE_FILES)
	$(COMPILER) $(OBJECT_DIR)/$@.o $(REVIEW_OBJECT_FILES) $(PERSUE_OBJECT_FILES) $(SECURE_OBJECT_FILES) $(LINKER_FLAGS) -o $(BINARY_DIR)/$@

# These are rules for compiling object files out of source files
$(OBJECT_DIR)/%.o: $(SOURCE_DIR)/*/%.c 
	$(COMPILER) $< -c $(COMPILE_FLAGS) -o $@

$(OBJECT_DIR)/%.o: $(SOURCE_DIR)/%.c 
	$(COMPILER) $< -c $(COMPILE_FLAGS) -o $@

# These are the executable programs that you can create
PROGRAMS := $(patsubst %.c, %, $(notdir $(wildcard $(SOURCE_DIR)/*.c)))

# These are the files that should be cleaned
DELETE_FILES := $(OBJECT_DIR)/*.o $(PROGRAMS)

# These are the files that should be kept (not automatically deleted)
.PRECIOUS: $(OBJECT_DIR)/%.o $(PROGRAMS)

$(CLEAN_TARGET):
	$(DELETE_CMD) $(DELETE_FLAGS) $(DELETE_FILES)

$(HELP_TARGET):
	@echo $(PROGRAMS) $(CLEAN_TARGET)

