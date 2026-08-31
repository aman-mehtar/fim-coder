-- fimcoder-113m in minuet-ai.nvim, against a local llama-server.
-- Drop into your lazy.nvim plugin spec. No API key: it is localhost.
--
-- The template is the important part. This model was trained with its own FIM
-- sentinels, so minuet must assemble the prompt itself rather than rely on a
-- provider default -- which is exactly what `template.prompt` is for.
return {
  'milanglacier/minuet-ai.nvim',
  dependencies = { 'nvim-lua/plenary.nvim' },
  config = function()
    require('minuet').setup {
      provider = 'openai_fim_compatible',
      -- One completion, not three. At 113M the second and third are rarely
      -- better and each costs a full generation.
      n_completions = 1,
      -- ~2600 tokens of the 4096 the model was trained for, leaving room for the
      -- suffix and the completion. Raise it if your GPU shrugs; lower it to ~4000
      -- characters if the first request in a file feels slow.
      context_window = 10000,
      context_ratio = 0.7,          -- favour the prefix over the suffix
      request_timeout = 3,
      throttle = 1000,              -- ms between requests
      debounce = 400,               -- ms after you stop typing
      notify = 'warn',
      provider_options = {
        openai_fim_compatible = {
          name = 'fimcoder',
          end_point = 'http://127.0.0.1:8080/v1/completions',
          model = 'fimcoder-113m',
          -- minuet insists on an api_key entry; any env var name works, and
          -- llama-server does not check it.
          api_key = function() return 'sk-no-key-needed' end,
          optional = {
            max_tokens = 64,
            temperature = 0.1,      -- near-greedy: this model rambles if you let it
            top_p = 0.9,
            stop = { '<|endoftext|>', '<|file_sep|>', '<|fim_prefix|>' },
          },
          template = {
            prompt = function(pref, suff)
              return '<|fim_prefix|>' .. pref .. '<|fim_suffix|>' .. suff
                     .. '<|fim_middle|>'
            end,
            suffix = false,         -- the prompt above already carries the suffix
          },
        },
      },
    }

    -- blink.cmp / nvim-cmp users: add 'minuet' as a source instead of this.
    vim.keymap.set('i', '<A-y>', function()
      require('minuet.virtualtext').action.accept()
    end, { desc = 'accept fimcoder completion' })
  end,
}
