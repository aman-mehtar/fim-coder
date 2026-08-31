-- Variant: feed the model repo context the way it was trained.
--
-- 29% of training documents were repo-level FIM -- <|repo_name|>, then sibling
-- files separated by <|file_sep|>, then the FIM triple on the target file. That
-- share of the data measurably paid off: repo-level FIM beat file-level on
-- first-line exact match (13.0% vs 9.6%). This template reproduces that shape
-- from your open buffers.
--
-- Replace the `template.prompt` in minuet.lua with this function.
prompt = function(pref, suff)
  local parts = { '<|repo_name|>' .. vim.fn.fnamemodify(vim.fn.getcwd(), ':t') }
  local cur = vim.api.nvim_get_current_buf()
  local budget = 6000                      -- characters of sibling context
  for _, b in ipairs(vim.api.nvim_list_bufs()) do
    local name = vim.api.nvim_buf_get_name(b)
    if budget > 0 and b ~= cur and name ~= ''
       and vim.api.nvim_buf_is_loaded(b) and vim.bo[b].buftype == '' then
      local body = table.concat(vim.api.nvim_buf_get_lines(b, 0, 150, false), '\n')
      body = body:sub(1, budget)
      budget = budget - #body
      table.insert(parts, '<|file_sep|>' .. vim.fn.fnamemodify(name, ':.') .. '\n' .. body)
    end
  end
  table.insert(parts, '<|file_sep|>' .. vim.fn.expand('%:.') .. '\n')
  return table.concat(parts) .. '<|fim_prefix|>' .. pref
         .. '<|fim_suffix|>' .. suff .. '<|fim_middle|>'
end
