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

all: master

%: $(OBJECT_DIR)/%.o $(SOURCE_DIR)/%.c
	$(COMPILER) $(OBJECT_DIR)/$@.o $(LINKER_FLAGS) -o $(BINARY_DIR)/$@

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

