(function () {
  'use strict';

  /* ---------------- 侧边栏折叠 / 移动端抽屉 ---------------- */
  var STORE_KEY = 'yq-open-groups';

  function readStore() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY) || '{}'); }
    catch (e) { return {}; }
  }
  function writeStore(v) {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(v)); } catch (e) {}
  }

  function initTree() {
    var store = readStore();
    var groups = document.querySelectorAll('.yq-tree-group');

    Array.prototype.forEach.call(groups, function (g) {
      var nameEl = g.querySelector('.yq-tree-group-name');
      var key = nameEl ? nameEl.textContent.trim() : '';
      // 当前文档所在分组始终展开；否则读取本地记忆
      if (!g.classList.contains('is-open') && store[key]) {
        g.classList.add('is-open');
      }

      var title = g.querySelector('.yq-tree-group-title');
      if (!title) return;
      var toggle = function () {
        g.classList.toggle('is-open');
        store[key] = g.classList.contains('is-open');
        writeStore(store);
      };
      title.addEventListener('click', toggle);
      title.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
      });
    });

    // 滚动到激活文档
    var active = document.querySelector('.yq-tree-doc.is-active');
    var sidebar = document.querySelector('.yq-sidebar');
    if (active && sidebar) {
      var offset = active.offsetTop - sidebar.clientHeight / 2;
      if (offset > 0) sidebar.scrollTop = offset;
    }
  }

  function initSidebarToggle() {
    var btn = document.getElementById('yqSidebarToggle');
    var sidebar = document.querySelector('.yq-sidebar');
    if (!btn || !sidebar) {
      if (btn) btn.style.display = 'none';
      return;
    }
    var mask = document.createElement('div');
    mask.className = 'yq-mask';
    document.body.appendChild(mask);

    function close() { sidebar.classList.remove('is-open'); mask.classList.remove('is-open'); }
    btn.addEventListener('click', function () {
      sidebar.classList.toggle('is-open');
      mask.classList.toggle('is-open');
    });
    mask.addEventListener('click', close);
  }

  /* ---------------- 右侧大纲 ---------------- */
  function initOutline() {
    var outline = document.getElementById('yqOutline');
    var content = document.querySelector('.yq-content');
    if (!outline || !content) return;

    var heads = content.querySelectorAll('h1, h2, h3');
    var items = [];
    var used = {};

    Array.prototype.forEach.call(heads, function (h) {
      var text = (h.textContent || '').trim();
      if (!text) return;
      if (!h.id) {
        var slug = text.toLowerCase().replace(/[^\w\u4e00-\u9fa5]+/g, '-').replace(/^-+|-+$/g, '');
        if (!slug) slug = 'section';
        if (used[slug]) { used[slug]++; slug = slug + '-' + used[slug]; } else { used[slug] = 1; }
        h.id = slug;
      }
      items.push({ id: h.id, text: text, level: parseInt(h.tagName.substring(1), 10), el: h });
    });

    if (items.length < 2) { outline.style.display = 'none'; return; }

    var minLevel = items.reduce(function (m, i) { return Math.min(m, i.level); }, 6);
    var html = '<div class="yq-outline-title">本文目录</div><ul>';
    items.forEach(function (i) {
      var lv = i.level - minLevel + 2; // 2,3,4
      if (lv > 4) lv = 4;
      html += '<li><a class="lv-' + lv + '" href="#' + i.id + '" data-target="' + i.id + '">' +
        i.text.replace(/</g, '&lt;') + '</a></li>';
    });
    html += '</ul>';
    outline.innerHTML = html;

    var links = outline.querySelectorAll('a');

    links = Array.prototype.slice.call(links);
    links.forEach(function (a) {
      a.addEventListener('click', function (e) {
        e.preventDefault();
        var target = document.getElementById(a.getAttribute('data-target'));
        if (!target) return;
        var top = target.getBoundingClientRect().top + window.pageYOffset - 72;
        window.scrollTo({ top: top, behavior: 'smooth' });
        history.replaceState(null, '', '#' + a.getAttribute('data-target'));
      });
    });

    var ticking = false;
    function spy() {
      var pos = window.pageYOffset + 100;
      var currentIndex = 0;
      for (var i = 0; i < items.length; i++) {
        if (items[i].el.offsetTop <= pos) currentIndex = i; else break;
      }
      links.forEach(function (a, idx) {
        if (idx === currentIndex) a.classList.add('is-active');
        else a.classList.remove('is-active');
      });
      ticking = false;
    }
    window.addEventListener('scroll', function () {
      if (!ticking) { ticking = true; window.requestAnimationFrame(spy); }
    });
    spy();
  }

  /* ---------------- 搜索 ---------------- */
  function initSearch() {
    var input = document.getElementById('yqSearchInput');
    var box = document.getElementById('yqSearchResult');
    if (!input || !box) return;

    var data = window.YQ_DOCS || [];

    function render(list, kw) {
      if (!list.length) {
        box.innerHTML = '<div class="yq-search-empty">没有找到相关文档</div>';
      } else {
        box.innerHTML = list.slice(0, 12).map(function (d) {
          return '<a href="' + d.url + '">' +
            '<div class="yq-sr-title">' + highlight(d.title, kw) + '</div>' +
            '<div class="yq-sr-book">' + d.book + (d.sub ? ' · ' + d.sub : '') + '</div>' +
            '</a>';
        }).join('');
      }
      box.classList.add('is-open');
    }

    function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }
    function highlight(text, kw) {
      var t = esc(text);
      if (!kw) return t;
      var i = t.toLowerCase().indexOf(kw.toLowerCase());
      if (i < 0) return t;
      return t.slice(0, i) + '<mark>' + t.slice(i, i + kw.length) + '</mark>' + t.slice(i + kw.length);
    }

    input.addEventListener('input', function () {
      var kw = input.value.trim().toLowerCase();
      if (!kw) { box.classList.remove('is-open'); return; }
      var list = data.filter(function (d) {
        return (d.title + ' ' + (d.sub || '') + ' ' + d.book + ' ' + (d.tags || '')).toLowerCase().indexOf(kw) >= 0;
      });
      render(list, input.value.trim());
    });

    input.addEventListener('focus', function () {
      if (input.value.trim()) box.classList.add('is-open');
    });

    document.addEventListener('click', function (e) {
      if (!box.contains(e.target) && e.target !== input) box.classList.remove('is-open');
    });

    document.addEventListener('keydown', function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); input.focus(); }
      if (e.key === 'Escape') { box.classList.remove('is-open'); input.blur(); }
    });
  }

  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }

  /* ---------------- 在线编辑（语雀式 + 同步 GitHub） ---------------- */
  function initEditor() {
    var cfgEl = document.getElementById('yqEditConfig');
    if (!cfgEl) return;
    var cfg;
    try { cfg = JSON.parse(cfgEl.textContent); } catch (e) { return; }
    if (!cfg || !cfg.enabled) return;

    var TOKEN_KEY = 'yq-github-token';
    var USER_KEY = 'yq-github-user';
    var siteBase = '';

    var editBtn = document.getElementById('yqEditBtn');
    var newBtn = document.getElementById('yqNewBtn');
    var modal = document.getElementById('yqEditModal');
    var tokenModal = document.getElementById('yqTokenModal');
    if (!editBtn || !modal || !tokenModal) return;

    var els = {
      file: document.getElementById('yqEditFile'),
      text: document.getElementById('yqEditText'),
      preview: document.getElementById('yqEditPreview'),
      status: document.getElementById('yqEditStatus'),
      save: document.getElementById('yqEditSave'),
      user: document.getElementById('yqEditUser'),
      logout: document.getElementById('yqEditLogout'),
      tokenInput: document.getElementById('yqTokenInput'),
      tokenOk: document.getElementById('yqTokenOk'),
      tokenStatus: document.getElementById('yqTokenStatus')
    };

    function api(path, opts) {
      opts = opts || {};
      var token = localStorage.getItem(TOKEN_KEY);
      var headers = { 'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json' };
      if (token) headers['Authorization'] = 'Bearer ' + token;
      return fetch('https://api.github.com' + path, {
        method: opts.method || 'GET',
        headers: headers,
        body: opts.body ? JSON.stringify(opts.body) : undefined
      });
    }

    function setStatus(el, msg, type) {
      el.textContent = msg || '';
      el.className = 'yq-edit-status' + (type ? ' is-' + type : '');
    }

    function allowedUsers() {
      return (cfg.allowed_users || []).map(function (u) { return String(u).toLowerCase(); });
    }

    // 验证 token 并比对白名单
    function verifyToken() {
      var token = localStorage.getItem(TOKEN_KEY);
      if (!token) return Promise.resolve(false);
      return api('/user').then(function (r) {
        if (!r.ok) { throw new Error('token'); }
        return r.json();
      }).then(function (u) {
        localStorage.setItem(USER_KEY, u.login);
        return allowedUsers().indexOf(String(u.login).toLowerCase()) !== -1
          ? u.login : false;
      }).catch(function () { return false; });
    }

    function showEditForUser(login) {
      editBtn.hidden = false;
      if (newBtn) newBtn.hidden = false;
      els.user.textContent = login;
    }

    // 启动时：若已有 token，验证是否在白名单
    verifyToken().then(function (ok) {
      if (ok) showEditForUser(ok);
    });

    /* ---- Token 弹窗 ---- */
    function openToken() {
      els.tokenInput.value = '';
      setStatus(els.tokenStatus, '');
      tokenModal.hidden = false;
      setTimeout(function () { els.tokenInput.focus(); }, 50);
    }
    function closeToken() { tokenModal.hidden = true; }

    els.tokenOk.addEventListener('click', function () {
      var t = els.tokenInput.value.trim();
      if (!t) { setStatus(els.tokenStatus, '请输入 Token', 'err'); return; }
      setStatus(els.tokenStatus, '验证中…');
      localStorage.setItem(TOKEN_KEY, t);
      verifyToken().then(function (ok) {
        if (!ok) {
          localStorage.removeItem(TOKEN_KEY);
          setStatus(els.tokenStatus, 'Token 无效，或该账号不在编辑白名单中', 'err');
          return;
        }
        closeToken();
        showEditForUser(ok);
        openEditor();
      }).catch(function () {
        localStorage.removeItem(TOKEN_KEY);
        setStatus(els.tokenStatus, '验证失败，请重试', 'err');
      });
    });

    els.logout.addEventListener('click', function () {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
      editBtn.hidden = true;
      els.user.textContent = '';
      setStatus(els.status, '');
    });

    /* ---- 编辑器 ---- */
    var currentPath = '';
    var currentSha = '';
    var previewTimer = null;

    function openEditor() {
      currentPath = editBtn.getAttribute('data-post-path');
      els.file.textContent = currentPath;
      setStatus(els.status, '正在加载原文…');
      modal.hidden = false;
      els.save.disabled = true;

      api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/' + encodeURIComponent(currentPath) + '?ref=' + cfg.branch)
        .then(function (r) {
          if (!r.ok) throw new Error('load');
          return r.json();
        })
        .then(function (data) {
          currentSha = data.sha;
          var content = decodeURIComponent(escape(window.atob(data.content.replace(/\s/g, ''))));
          els.text.value = content;
          setStatus(els.status, '已加载', 'ok');
          els.save.disabled = false;
          renderPreview();
        })
        .catch(function () {
          setStatus(els.status, '加载原文失败（无权限或网络问题）', 'err');
        });
    }

    function closeEditor() { modal.hidden = true; }

    function renderPreview() {
      var md = els.text.value;
      els.preview.innerHTML = '<div class="yq-preview-empty">渲染中…</div>';
      // 优先使用 GitHub 渲染接口，效果最贴近语雀
      fetch('https://api.github.com/markdown', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: md, mode: 'gfm', context: cfg.owner + '/' + cfg.repo })
      }).then(function (r) {
        if (!r.ok) throw new Error('preview');
        return r.text();
      }).then(function (html) {
        els.preview.innerHTML = html;
        if (window.MathJax) {
          try { window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub, els.preview]); } catch (e) {}
        }
      }).catch(function () {
        var div = document.createElement('div');
        div.className = 'yq-preview-empty';
        div.textContent = '预览不可用（可能触发 GitHub 频率限制），保存后线上可见效果。';
        els.preview.innerHTML = '';
        els.preview.appendChild(div);
      });
    }

    els.text.addEventListener('input', function () {
      clearTimeout(previewTimer);
      previewTimer = setTimeout(renderPreview, 400);
    });

    els.save.addEventListener('click', function () {
      if (!currentPath || !currentSha) return;
      setStatus(els.status, '保存中…');
      els.save.disabled = true;
      var doPut = function (sha) {
        return api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/' + encodeURIComponent(currentPath), {
          method: 'PUT',
          body: {
            message: 'docs(' + currentPath + '): update via online editor',
            content: window.btoa(unescape(encodeURIComponent(els.text.value))),
            sha: sha,
            branch: cfg.branch
          }
        });
      };
      doPut(currentSha).then(function (r) {
        if (r.ok) return r.json();
        // sha 冲突（409）时自动重新拉取最新 sha 重试一次
        if (r.status === 409) {
          return api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/' + encodeURIComponent(currentPath) + '?ref=' + cfg.branch)
            .then(function (rr) { return rr.json(); })
            .then(function (d) { currentSha = d.sha; return doPut(d.sha); });
        }
        return r.json().then(function (body) {
          var msg = (body && body.message) ? body.message : ('HTTP ' + r.status);
          throw new Error('保存失败（GitHub: ' + msg + '）');
        }, function () { throw new Error('保存失败（HTTP ' + r.status + '）'); });
      }).then(function (data) {
        if (data && data.content && data.content.sha) currentSha = data.content.sha;
        setStatus(els.status, '已同步到 GitHub ✓ 稍后站点重新构建即可见', 'ok');
        setTimeout(closeEditor, 1400);
      }).catch(function (err) {
        setStatus(els.status, (err && err.message) ? err.message : '保存失败（权限不足或冲突，请重试）', 'err');
        els.save.disabled = false;
      });
    });

    /* ---- 打开流程：有 token 直接开，无则先要 token ---- */
    editBtn.addEventListener('click', function () {
      verifyToken().then(function (ok) {
        if (!ok) { openToken(); return; }
        showEditForUser(ok);
        openEditor();
      });
    });

    // 关闭事件（背景 / ✕ / 取消）
    Array.prototype.forEach.call(modal.querySelectorAll('[data-close]'), function (el) {
      el.addEventListener('click', closeEditor);
    });
    Array.prototype.forEach.call(tokenModal.querySelectorAll('[data-token-close]'), function (el) {
      el.addEventListener('click', closeToken);
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { closeEditor(); closeToken(); }
    });

    /* ============ 新建文档 ============ */
    var newModal = document.getElementById('yqNewModal');
    var delModal = document.getElementById('yqDelModal');
    var booksData = [];
    try {
      var bd = document.getElementById('yqBooksData');
      if (bd) booksData = JSON.parse(bd.textContent);
    } catch (e) {}

    var slugify = function (s) {
      return String(s).trim().toLowerCase()
        .replace(/[^\w\u4e00-\u9fa5]+/g, '-')
        .replace(/^-+|-+$/g, '') || 'untitled';
    };

    function fillBookOptions() {
      var sel = document.getElementById('yqNewBook');
      var gsel = document.getElementById('yqNewGroup');
      sel.innerHTML = '';
      gsel.innerHTML = '';
      booksData.forEach(function (b, i) {
        var o = document.createElement('option');
        o.value = i;
        o.textContent = (b.icon ? b.icon + ' ' : '') + b.title;
        sel.appendChild(o);
      });
      fillGroupOptions();
      sel.onchange = fillGroupOptions;
    }
    function fillGroupOptions() {
      var sel = document.getElementById('yqNewBook');
      var gsel = document.getElementById('yqNewGroup');
      var b = booksData[sel.value];
      gsel.innerHTML = '';
      if (!b) return;
      (b.groups || []).forEach(function (g, i) {
        var o = document.createElement('option');
        o.value = i;
        o.textContent = g.title;
        gsel.appendChild(o);
      });
    }

    function openNew() {
      document.getElementById('yqNewTitle').value = '';
      document.getElementById('yqNewText').value = '';
      setStatus(document.getElementById('yqNewStatus'), '');
      fillBookOptions();
      newModal.hidden = false;
      setTimeout(function () { document.getElementById('yqNewTitle').focus(); }, 50);
    }
    function closeNew() { newModal.hidden = true; }

    document.getElementById('yqNewSave').addEventListener('click', function () {
      var titleEl = document.getElementById('yqNewTitle');
      var textEl = document.getElementById('yqNewText');
      var statusEl = document.getElementById('yqNewStatus');
      var title = titleEl.value.trim();
      if (!title) { setStatus(statusEl, '请填写标题', 'err'); return; }

      var b = booksData[document.getElementById('yqNewBook').value];
      var g = (b.groups || [])[document.getElementById('yqNewGroup').value];
      var prefix = (g && g.prefix && g.prefix[0]) ? g.prefix[0] + '_' : '';
      var tag = (g && g.tag && g.tag[0]) ? g.tag[0] : b.title;
      var date = new Date();
      var ymd = date.getFullYear() + '-' +
        ('0' + (date.getMonth() + 1)).slice(-2) + '-' +
        ('0' + date.getDate()).slice(-2);
      var slug = prefix + slugify(title);
      var path = '_posts/' + ymd + '-' + slug + '.markdown';

      var body = '---\n' +
        'layout: post\n' +
        'title: "' + title.replace(/"/g, '\\"') + '"\n' +
        'subtitle: ""\n' +
        'date: ' + ymd + ' ' +
        ('0' + date.getHours()).slice(-2) + ':' +
        ('0' + date.getMinutes()).slice(-2) + ':' +
        ('0' + date.getSeconds()).slice(-2) + '\n' +
        'author: "' + (localStorage.getItem(USER_KEY) || cfg.owner) + '"\n' +
        'tags: [' + tag + ']\n' +
        '---\n\n' + textEl.value + '\n';

      setStatus(statusEl, '创建中…');
      this.disabled = true;
      api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/' + encodeURIComponent(path), {
        method: 'POST',
        body: {
          message: 'docs: new post ' + path,
          content: window.btoa(unescape(encodeURIComponent(body))),
          branch: cfg.branch
        }
      }).then(function (r) {
        if (!r.ok) throw new Error('create');
        return r.json();
      }).then(function (data) {
        setStatus(statusEl, '已创建 ✓ 正在跳转…', 'ok');
        setTimeout(function () {
          window.location.href = (siteBase || '') + '/' + data.content.path
            .replace(/^_posts\//, '')
            .replace(/\.markdown$/, '')
            .replace(/(\d{4})-(\d{2})-(\d{2})-(.*)/, '$1/$2/$3/$4/');
        }, 1000);
      }).catch(function () {
        setStatus(statusEl, '创建失败（文件名可能已存在或权限不足）', 'err');
        this.disabled = false;
      }.bind(this));
    });

    if (newBtn) {
      newBtn.addEventListener('click', function () {
        verifyToken().then(function (ok) {
          if (!ok) { openToken(); return; }
          showEditForUser(ok);
          openNew();
        });
      });
    }
    Array.prototype.forEach.call(newModal.querySelectorAll('[data-new-close]'), function (el) {
      el.addEventListener('click', closeNew);
    });

    /* ============ 删除文档 ============ */
    var delPath = '';
    var delName = '';
    function openDelete() {
      delPath = editBtn.getAttribute('data-post-path');
      delName = delPath.split('/').pop();
      document.getElementById('yqDelName').textContent = delName;
      setStatus(document.getElementById('yqDelStatus'), '');
      delModal.hidden = false;
    }
    function closeDelete() { delModal.hidden = true; }

    document.getElementById('yqEditDelete').addEventListener('click', function () {
      closeEditor();
      openDelete();
    });
    document.getElementById('yqDelOk').addEventListener('click', function () {
      var statusEl = document.getElementById('yqDelStatus');
      setStatus(statusEl, '删除中…');
      this.disabled = true;
      api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/' + encodeURIComponent(delPath) + '?ref=' + cfg.branch)
        .then(function (r) { if (!r.ok) throw new Error('get'); return r.json(); })
        .then(function (data) {
          return api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/' + encodeURIComponent(delPath), {
            method: 'DELETE',
            body: {
              message: 'docs: delete ' + delPath,
              sha: data.sha,
              branch: cfg.branch
            }
          });
        })
        .then(function (r) { if (!r.ok) throw new Error('del'); return r; })
        .then(function () {
          setStatus(statusEl, '已删除 ✓ 即将返回首页', 'ok');
          setTimeout(function () { window.location.href = (siteBase || '') + '/'; }, 1000);
        })
        .catch(function () {
          setStatus(statusEl, '删除失败（权限不足或文件已被改动）', 'err');
          this.disabled = false;
        }.bind(this));
    });
    Array.prototype.forEach.call(delModal.querySelectorAll('[data-del-close]'), function (el) {
      el.addEventListener('click', closeDelete);
    });
  }

  /* ---------------- 删除知识库（一键删除父目录及全部文章） ---------------- */
  function initBookDelete() {
    var cfgEl = document.getElementById('yqEditConfig');
    if (!cfgEl) return;
    var cfg;
    try { cfg = JSON.parse(cfgEl.textContent); } catch (e) { return; }
    if (!cfg || !cfg.enabled) return;

    var TOKEN_KEY = 'yq-github-token';
    var USER_KEY = 'yq-github-user';

    var btn = document.getElementById('yqBookDelBtn');
    var modal = document.getElementById('yqBookDelModal');
    if (!btn || !modal) return;

    var slugMeta = document.querySelector('meta[name="yq-book-slug"]');
    var titleMeta = document.querySelector('meta[name="yq-book-title"]');
    var pathsMeta = document.querySelector('meta[name="yq-book-paths"]');
    if (!slugMeta || !titleMeta || !pathsMeta) return;

    var slug = slugMeta.content;
    var bookTitle = titleMeta.content;
    var postPaths = pathsMeta.content.split(',').map(function (s) { return s.trim(); }).filter(Boolean);

    function api(path, opts) {
      opts = opts || {};
      var token = localStorage.getItem(TOKEN_KEY);
      var headers = { 'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json' };
      if (token) headers['Authorization'] = 'Bearer ' + token;
      return fetch('https://api.github.com' + path, {
        method: opts.method || 'GET',
        headers: headers,
        body: opts.body ? JSON.stringify(opts.body) : undefined
      });
    }
    function setStatus(el, msg, type) {
      el.textContent = msg || '';
      el.className = 'yq-edit-status' + (type ? ' is-' + type : '');
    }
    function verifyToken() {
      var token = localStorage.getItem(TOKEN_KEY);
      if (!token) return Promise.resolve(false);
      return api('/user').then(function (r) {
        if (!r.ok) throw new Error('token');
        return r.json();
      }).then(function (u) {
        localStorage.setItem(USER_KEY, u.login);
        return (cfg.allowed_users || []).map(function (x) { return String(x).toLowerCase(); })
          .indexOf(String(u.login).toLowerCase()) !== -1 ? u.login : false;
      }).catch(function () { return false; });
    }

    var okBtn = document.getElementById('yqBookDelOk');
    var inputEl = document.getElementById('yqBookDelInput');
    var statusEl = document.getElementById('yqBookDelStatus');

    // 启动时若已有有效 token，直接显示删除按钮
    verifyToken().then(function (ok) {
      if (ok) btn.hidden = false;
    });

    function openModal() {
      document.getElementById('yqBookDelName').textContent = bookTitle;
      document.getElementById('yqBookDelConfirmName').textContent = bookTitle;
      document.getElementById('yqBookDelCount').textContent = postPaths.length;
      inputEl.value = '';
      okBtn.disabled = true;
      setStatus(statusEl, '');
      modal.hidden = false;
      setTimeout(function () { inputEl.focus(); }, 50);
    }
    function closeModal() { modal.hidden = true; }

    inputEl.addEventListener('input', function () {
      okBtn.disabled = (inputEl.value.trim() !== bookTitle);
    });

    btn.addEventListener('click', function () {
      verifyToken().then(function (ok) {
        if (!ok) {
          // 复用编辑的 token 弹窗机制：无 token 时提示
          setStatus(statusEl, '请先在任一篇文章点"编辑"并验证 Token 后再删除知识库', 'err');
          return;
        }
        openModal();
      });
    });
    okBtn.addEventListener('click', function () {
      if (okBtn.disabled) return;
      okBtn.disabled = true;
      setStatus(statusEl, '准备删除…');

      // 1) 逐个删除文章文件（先取 sha 再 DELETE）
      var delOne = function (path) {
        return api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/' + encodeURIComponent(path) + '?ref=' + cfg.branch)
          .then(function (r) { if (!r.ok) throw new Error('get ' + path); return r.json(); })
          .then(function (data) {
            return api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/' + encodeURIComponent(path), {
              method: 'DELETE',
              body: { message: 'docs: delete book ' + slug + ' -> ' + path, sha: data.sha, branch: cfg.branch }
            });
          })
          .then(function (r) { if (!r.ok) throw new Error('del ' + path); return r; });
      };

      var chain = Promise.resolve();
      postPaths.forEach(function (p) {
        chain = chain.then(function () { return delOne(p); });
      });

      // 2) 读取 books.yml，删除该 book 块后写回
      chain = chain.then(function () {
        setStatus(statusEl, '正在更新目录配置…');
        return api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/_data/books.yml?ref=' + cfg.branch)
          .then(function (r) { if (!r.ok) throw new Error('get yml'); return r.json(); })
          .then(function (data) {
            var yml = decodeURIComponent(escape(window.atob(data.content.replace(/\s/g, ''))));
            var newYml = removeBookBlock(yml, slug);
            return api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/_data/books.yml', {
              method: 'PUT',
              body: {
                message: 'docs: remove book ' + slug + ' from books.yml',
                content: window.btoa(unescape(encodeURIComponent(newYml))),
                sha: data.sha,
                branch: cfg.branch
              }
            }).then(function (r) { if (!r.ok) throw new Error('put yml'); return r; });
          });
      });

      // 3) 删除 book/<slug>.html
      chain = chain.then(function () {
        setStatus(statusEl, '正在移除知识库页面…');
        var bp = 'book/' + slug + '.html';
        return api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/' + encodeURIComponent(bp) + '?ref=' + cfg.branch)
          .then(function (r) { if (!r.ok) return; return r.json(); })
          .then(function (data) {
            if (!data || !data.sha) return;
            return api('/repos/' + cfg.owner + '/' + cfg.repo + '/contents/' + encodeURIComponent(bp), {
              method: 'DELETE',
              body: { message: 'docs: remove book page ' + slug, sha: data.sha, branch: cfg.branch }
            }).then(function (r) { if (!r.ok) throw new Error('del page'); return r; });
          });
      });

      chain.then(function () {
        setStatus(statusEl, '知识库已删除 ✓ 即将返回首页', 'ok');
        setTimeout(function () { window.location.href = '/'; }, 1200);
      }).catch(function (err) {
        setStatus(statusEl, '删除中断：' + (err && err.message ? err.message : '未知错误') + '（部分文件可能已删除，请检查仓库）', 'err');
        okBtn.disabled = false;
      });
    });

    Array.prototype.forEach.call(modal.querySelectorAll('[data-bookdel-close]'), function (el) {
      el.addEventListener('click', closeModal);
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') closeModal();
    });
  }

  // 按 slug 从 books.yml 中删除整个 book 块（每个块以 "- slug:" 起始）
  function removeBookBlock(yml, slug) {
    var lines = yml.split(/\n/);
    var keep = [], i = 0, skip = false;
    var target = '- slug: ' + slug;
    while (i < lines.length) {
      var line = lines[i];
      if (/^\s*-\s*slug\s*:\s*\S/.test(line)) {
        // 新块开始
        if (line.replace(/^\s*-\s*slug\s*:\s*/, '').trim() === slug) {
          skip = true; // 跳过此块
        } else {
          skip = false;
        }
      }
      if (!skip) keep.push(line);
      i++;
    }
    return keep.join('\n');
  }

  /* ---------------- 划线评论（匿名，存 GitHub Issues） ---------------- */
  function initComments() {
    var cfgEl = document.getElementById('yqEditConfig');
    if (!cfgEl) return;
    var cfg;
    try { cfg = JSON.parse(cfgEl.textContent); } catch (e) { return; }
    if (!cfg || !cfg.enabled) return;

    var content = document.getElementById('yqContent');
    var listEl = document.getElementById('yqCommentList');
    var countEl = document.getElementById('yqCommentCount');
    var selPop = document.getElementById('yqSelPop');
    var cmtModal = document.getElementById('yqCmtModal');
    if (!content || !listEl || !cmtModal) return;

    var TOKEN_KEY = 'yq-github-token';
    var LABEL_PREFIX = (cfg.comments && cfg.comments.label_prefix) || 'post:';
    var pm = document.querySelector('meta[name="yq-post-path"]');
    var postPath = pm ? pm.content : location.pathname;
    var label = LABEL_PREFIX + postPath;

    /* 匿名昵称 */
    var ANON_KEY = 'yq-anon-name';
    function anonName() {
      var n = localStorage.getItem(ANON_KEY);
      if (!n) {
        var pool = '甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未';
        n = '匿名' + pool[Math.floor(Math.random() * pool.length)] +
            (Math.floor(Math.random() * 90) + 10);
        localStorage.setItem(ANON_KEY, n);
      }
      return n;
    }

    /* GitHub API */
    function api(path, opts) {
      opts = opts || {};
      var token = localStorage.getItem(TOKEN_KEY);
      var headers = { 'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json' };
      if (token) headers['Authorization'] = 'Bearer ' + token;
      return fetch('https://api.github.com' + path, {
        method: opts.method || 'GET',
        headers: headers,
        body: opts.body ? JSON.stringify(opts.body) : undefined
      });
    }

    /* ---- 文本偏移工具 ---- */
    // 计算 (node, offset) 在 content 纯文本中的全局偏移
    function getOffset(node, offset) {
      var walk = document.createTreeWalker(content, NodeFilter.SHOW_TEXT, null);
      var total = 0, n;
      while ((n = walk.nextNode())) {
        if (n === node) return total + offset;
        total += n.nodeValue.length;
      }
      return total + offset;
    }
    function getText() {
      return content.innerText;
    }

    /* 解析评论 body 中的 yq-comment JSON */
    function parseComment(body) {
      var m = body.match(/<!--\s*yq-comment\s*([\s\S]*?)\s*-->/);
      if (!m) return null;
      try { return JSON.parse(m[1]); } catch (e) { return null; }
    }

    /* ---- Issue 管理 ---- */
    function findIssue() {
      return api('/repos/' + cfg.owner + '/' + cfg.repo + '/issues?labels=' +
        encodeURIComponent(label) + '&state=all&per_page=1')
        .then(function (r) { return r.json(); })
        .then(function (arr) { return (arr && arr.length) ? arr[0] : null; });
    }
    function createIssue() {
      var title = document.title.split(' - ')[0] || 'comment';
      return api('/repos/' + cfg.owner + '/' + cfg.repo + '/issues', {
        method: 'POST',
        body: {
          title: '[评论] ' + title,
          body: '文章：' + location.href + '\n路径：' + postPath,
          labels: [label]
        }
      }).then(function (r) { return r.json(); });
    }

    /* ---- 高亮渲染 ---- */
    function renderHighlights(comments) {
      // 清除旧高亮
      var old = content.querySelectorAll('mark.yq-hl');
      Array.prototype.forEach.call(old, function (m) {
        var p = m.parentNode;
        p.replaceChild(document.createTextNode(m.textContent), m);
        p.normalize();
      });
      var ranges = comments
        .filter(function (c) { return typeof c.start === 'number' && typeof c.end === 'number'; })
        .sort(function (a, b) { return a.start - b.start; });
      if (!ranges.length) return;

      var text = getText();
      var walk = document.createTreeWalker(content, NodeFilter.SHOW_TEXT, null);
      var nodes = [], n, total = 0;
      while ((n = walk.nextNode())) { nodes.push({ node: n, start: total, len: n.nodeValue.length }); total += n.nodeValue.length; }

      ranges.forEach(function (c) {
        if (c.end > total) c.end = total;
        nodes.forEach(function (nd) {
          var ns = nd.start, ne = nd.start + nd.len;
          // 该文本节点与评论区间的交集
          var a = Math.max(ns, c.start), b = Math.min(ne, c.end);
          if (a >= b) return;
          var node = nd.node;
          var frag = document.createDocumentFragment();
          if (a > ns) frag.appendChild(document.createTextNode(node.nodeValue.slice(0, a - ns)));
          var mark = document.createElement('mark');
          mark.className = 'yq-hl' + (c.mine ? ' is-mine' : '');
          mark.dataset.cid = c.id || '';
          mark.textContent = node.nodeValue.slice(a - ns, b - ns);
          frag.appendChild(mark);
          if (b < ne) frag.appendChild(document.createTextNode(node.nodeValue.slice(b - ns)));
          node.parentNode.replaceChild(frag, node);
          // 节点已替换，更新后续引用
          nd.node = frag.childNodes[0] ? null : nd.node;
        });
      });
    }

    /* ---- 评论列表渲染 ---- */
    function renderList(comments) {
      listEl.innerHTML = '';
      countEl.textContent = comments.length;
      if (!comments.length) {
        listEl.innerHTML = '<div class="yq-comment-empty">还没有评论，选中正文文字即可划线评论。</div>';
        return;
      }
      comments.forEach(function (c) {
        var item = document.createElement('div');
        item.className = 'yq-comment-item' + (c.mine ? ' is-mine' : '');
        var d = c.ts ? new Date(c.ts) : null;
        var date = d ? d.toLocaleString('zh-CN', { hour12: false }) : '';
        item.innerHTML =
          '<div class="yq-comment-meta"><span class="yq-comment-author"></span>' +
          '<span>' + date + '</span></div>' +
          (c.quote ? '<blockquote class="yq-comment-quote"></blockquote>' : '') +
          '<div class="yq-comment-text"></div>';
        item.querySelector('.yq-comment-author').textContent = c.author || '匿名';
        if (c.quote) item.querySelector('.yq-comment-quote').textContent = c.quote;
        item.querySelector('.yq-comment-text').textContent = c.text || '';
        item.addEventListener('mouseenter', function () {
          var m = content.querySelector('mark.yq-hl[data-cid="' + (c.id || '') + '"]');
          if (m) m.classList.add('is-active');
        });
        item.addEventListener('mouseleave', function () {
          var m = content.querySelector('mark.yq-hl[data-cid="' + (c.id || '') + '"]');
          if (m) m.classList.remove('is-active');
        });
        listEl.appendChild(item);
      });
    }

    /* ---- 加载评论 ---- */
    var issueNumber = null;
    function loadComments() {
      return findIssue().then(function (issue) {
        if (!issue) { renderList([]); renderHighlights([]); return []; }
        issueNumber = issue.number;
        return api('/repos/' + cfg.owner + '/' + cfg.repo + '/issues/' + issue.number + '/comments')
          .then(function (r) { return r.json(); })
          .then(function (arr) {
            var me = anonName();
            var comments = (arr || []).map(function (c, i) {
              var p = parseComment(c.body);
              if (!p) return null;
              p.id = 'c' + issue.number + '_' + i;
              p.mine = (p.author === me);
              return p;
            }).filter(Boolean);
            renderList(comments);
            renderHighlights(comments);
            return comments;
          });
      });
    }

    /* ---- 选区 → 弹钮 ---- */
    var pendingSel = null;
    function onSelection() {
      var sel = window.getSelection();
      if (!sel || sel.isCollapsed || sel.rangeCount === 0) { selPop.hidden = true; return; }
      var range = sel.getRangeAt(0);
      if (!content.contains(range.commonAncestorContainer)) { selPop.hidden = true; return; }
      var start = getOffset(range.startContainer, range.startOffset);
      var end = getOffset(range.endContainer, range.endOffset);
      if (end < start) { var t = start; start = end; end = t; }
      if (end === start) { selPop.hidden = true; return; }
      pendingSel = { start: start, end: end, quote: sel.toString() };
      var rect = range.getBoundingClientRect();
      selPop.hidden = false;
      selPop.style.left = (rect.left + rect.width / 2 + window.scrollX) + 'px';
      selPop.style.top = (rect.top + window.scrollY - 8) + 'px';
    }
    document.addEventListener('mouseup', function () { setTimeout(onSelection, 10); });
    document.addEventListener('selectionchange', function () {
      var sel = window.getSelection();
      if (!sel || sel.isCollapsed) selPop.hidden = true;
    });

    document.getElementById('yqSelComment').addEventListener('click', function () {
      if (!pendingSel) return;
      selPop.hidden = true;
      window.getSelection().removeAllRanges();
      document.getElementById('yqCmtQuote').textContent = pendingSel.quote;
      document.getElementById('yqCmtText').value = '';
      setCmtStatus('');
      cmtModal.hidden = false;
      setTimeout(function () { document.getElementById('yqCmtText').focus(); }, 50);
    });

    function setCmtStatus(msg, type) {
      var el = document.getElementById('yqCmtStatus');
      el.textContent = msg || '';
      el.className = 'yq-edit-status' + (type ? ' is-' + type : '');
    }

    document.getElementById('yqCmtSave').addEventListener('click', function () {
      var text = document.getElementById('yqCmtText').value.trim();
      if (!text) { setCmtStatus('请填写评论内容', 'err'); return; }
      if (!localStorage.getItem(TOKEN_KEY)) {
        setCmtStatus('需要填写 GitHub Token 才能发表（评论匿名，仅用于写入 Issues）', 'err');
        return;
      }
      setCmtStatus('发表中…');
      this.disabled = true;
      var payload = {
        quote: pendingSel.quote,
        text: text,
        start: pendingSel.start,
        end: pendingSel.end,
        author: anonName(),
        ts: Date.now()
      };
      var body = '<!--yq-comment\n' + JSON.stringify(payload) + '\n-->\n';

      function post() {
        return api('/repos/' + cfg.owner + '/' + cfg.repo + '/issues/' + issueNumber + '/comments', {
          method: 'POST', body: { body: body }
        });
      }
      var chain = issueNumber
        ? Promise.resolve()
        : createIssue().then(function (iss) { issueNumber = iss.number; });

      chain.then(post)
        .then(function (r) { if (!r.ok) throw new Error('post'); return r; })
        .then(function () { return loadComments(); })
        .then(function () {
          setCmtStatus('已发表 ✓', 'ok');
          setTimeout(function () { cmtModal.hidden = true; }, 800);
        })
        .catch(function () {
          setCmtStatus('发表失败（Token 无权限或网络问题）', 'err');
          this.disabled = false;
        }.bind(this));
    });

    Array.prototype.forEach.call(cmtModal.querySelectorAll('[data-cmt-close]'), function (el) {
      el.addEventListener('click', function () { cmtModal.hidden = true; });
    });

    loadComments();
  }

  ready(function () {
    initTree();
    initSidebarToggle();
    initOutline();
    initSearch();
    initEditor();
    initBookDelete();
    initComments();
  });
})();
