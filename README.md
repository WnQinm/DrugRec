data:

- entity, desc
- link (head, link, tail)

pre-process:

- entity passage(pos, k\*neg) sim_entity(k)
- head-id link tail-id (以link的index作为Dataset索引)
- head(pos, k\*neg, passage(pos, k\*neg)) link tail(pos, k\*neg, passage(pos, k\*neg))

loss:

- entity reconstruction cross entropy(entity, passage)
- infoNCE(head+link, tail, neg_tail)  (bs, embed_dim),(bs, embed_dim),(bs, k, embed_dim)
- infoNCE(tail-link, head, neg_head)

$$
-\log\frac{e^{s(q)/\tau}}{e^{s(q+)/\tau}+\sum e^{s(q-)/\tau}}
$$
